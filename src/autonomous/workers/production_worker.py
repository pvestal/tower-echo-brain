"""
Production Worker — Autonomous anime-studio pipeline management.

Runs every 10 minutes. Each cycle:
1. Check anime-studio health
2. Read orchestrator status + pipeline phases
3. Auto-approve high-quality keyframes (CLIP score >= 0.75)
4. Flag borderline images for manual review via Telegram
5. Ensure orchestrator is enabled during active hours
6. Report stalls and blockers via Telegram
7. Track progress across projects and send milestone alerts
"""
import logging
import os
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Pacific time offset (simplified — doesn't handle DST edge)
PT_OFFSET = timedelta(hours=-7)  # PDT

# Auto-approve threshold: images with quality_score >= this are approved without review
AUTO_APPROVE_THRESHOLD = 0.75
# Below this, auto-reject
AUTO_REJECT_THRESHOLD = 0.30
# Between reject and approve thresholds → flag for manual review


class ProductionWorker:
    """Autonomous production pipeline monitor and actor."""

    def __init__(self):
        self._last_report: dict[int, dict] = {}  # project_id → last known state
        self._stall_counts: dict[int, int] = {}  # project_id → consecutive stall ticks
        self._cycle_count = 0

    async def run_cycle(self):
        self._cycle_count += 1
        try:
            from src.integrations.anime_studio_client import anime_studio
        except ImportError:
            logger.error("anime_studio_client not available")
            return

        # 1. Health check
        healthy = await anime_studio.health()
        if not healthy:
            logger.warning("Anime Studio unreachable — skipping cycle")
            if self._cycle_count % 6 == 0:  # Alert every hour
                await self._notify("Anime Studio has been unreachable for ~1 hour")
            return

        # 2. Get orchestrator status
        orch_status = await anime_studio.orchestrator_status()
        if not orch_status:
            return

        orch_enabled = orch_status.get("enabled", False)

        # 3. Auto-enable during active hours if disabled
        # DISABLED: Let the overnight timers (tower-overnight-start/stop) handle
        # orchestrator scheduling. This worker was fighting manual disables.
        # now_pt = datetime.now(timezone.utc) + PT_OFFSET
        # hour_pt = now_pt.hour
        # in_active_window = hour_pt >= 18 or hour_pt < 6
        # if in_active_window and not orch_enabled:
        #     logger.info("Active hours + orchestrator disabled → enabling")
        #     await anime_studio.orchestrator_toggle(True)
        #     await self._notify("Orchestrator auto-enabled for overnight production window")

        # 4. Process each project
        projects = await anime_studio.list_projects()
        for project in projects:
            pid = project.get("id")
            pname = project.get("name", f"Project {pid}")
            if not pid:
                continue
            await self._process_project(anime_studio, pid, pname)

        # 5. Periodic summary (every 6 cycles = ~1 hour)
        if self._cycle_count % 6 == 0:
            await self._send_summary(projects)

    async def _process_project(self, client, project_id: int, project_name: str):
        """Process a single project: auto-approve, detect stalls, report progress."""

        # Auto-approve pending images
        try:
            pending = await client.pending_images(project_id)
            if isinstance(pending, list) and pending:
                approved, rejected, flagged = 0, 0, 0
                for img in pending:
                    score = None
                    meta = img.get("metadata") or {}
                    score = meta.get("quality_score") or img.get("quality_score")
                    if score is None:
                        # Check vision_review sub-dict
                        vr = meta.get("vision_review") or {}
                        score = vr.get("overall_score") or vr.get("quality_score")

                    img_id = img.get("id") or img.get("image_id")
                    if not img_id:
                        continue

                    if score is not None:
                        if score >= AUTO_APPROVE_THRESHOLD:
                            await client.approve_image(img_id, True)
                            approved += 1
                        elif score < AUTO_REJECT_THRESHOLD:
                            await client.approve_image(img_id, False)
                            rejected += 1
                        else:
                            flagged += 1
                    # No score → leave pending for manual review

                if approved or rejected:
                    logger.info(
                        "%s: auto-approved %d, auto-rejected %d, flagged %d for review",
                        project_name, approved, rejected, flagged,
                    )
                if flagged > 0:
                    await self._notify(
                        f"{project_name}: {flagged} images need manual review "
                        f"(score {AUTO_REJECT_THRESHOLD}-{AUTO_APPROVE_THRESHOLD})"
                    )
        except Exception as e:
            logger.debug("Pending images check failed for %s: %s", project_name, e)

        # Check for stalls — compare to last known state
        try:
            pipeline = await client.pipeline_status(project_id)
            if pipeline:
                current_phase = self._extract_phase(pipeline)
                last = self._last_report.get(project_id, {})
                last_phase = last.get("phase")

                if current_phase and current_phase == last_phase:
                    self._stall_counts[project_id] = self._stall_counts.get(project_id, 0) + 1
                    if self._stall_counts[project_id] == 6:  # ~1 hour stall
                        await self._notify(
                            f"{project_name}: stalled on '{current_phase}' for ~1 hour"
                        )
                else:
                    if last_phase and current_phase and current_phase != last_phase:
                        await self._notify(
                            f"{project_name}: advanced {last_phase} → {current_phase}"
                        )
                    self._stall_counts[project_id] = 0

                self._last_report[project_id] = {
                    "phase": current_phase,
                    "timestamp": datetime.utcnow().isoformat(),
                }
        except Exception as e:
            logger.debug("Pipeline status check failed for %s: %s", project_name, e)

    def _extract_phase(self, pipeline_data) -> str | None:
        """Extract current phase from pipeline status response."""
        if isinstance(pipeline_data, dict):
            # Try common response shapes
            phases = pipeline_data.get("phases") or pipeline_data.get("pipeline") or []
            if isinstance(phases, list):
                for p in phases:
                    if isinstance(p, dict) and p.get("status") == "active":
                        return p.get("phase") or p.get("name")
            # Flat dict with current_phase
            return pipeline_data.get("current_phase") or pipeline_data.get("phase")
        return None

    async def _send_summary(self, projects: list[dict]):
        """Send hourly production summary via Telegram."""
        lines = ["Production Status:"]
        for p in projects[:10]:  # Top 10
            pid = p.get("id")
            name = p.get("name", "?")
            state = self._last_report.get(pid, {})
            phase = state.get("phase", "unknown")
            stalls = self._stall_counts.get(pid, 0)
            stall_tag = f" (stalled {stalls * 10}min)" if stalls >= 3 else ""
            lines.append(f"  {name}: {phase}{stall_tag}")

        summary = "\n".join(lines)
        logger.info(summary)
        # Only send Telegram summary during active hours
        now_pt = datetime.now(timezone.utc) + PT_OFFSET
        if now_pt.hour >= 18 or now_pt.hour < 6:
            await self._notify(summary)

    async def _notify(self, message: str):
        """Send notification via Telegram."""
        try:
            from src.integrations.mcp_service import mcp_service
            await mcp_service.send_notification(
                message=message,
                channel="telegram",
            )
        except Exception as e:
            logger.debug("Notification failed: %s", e)

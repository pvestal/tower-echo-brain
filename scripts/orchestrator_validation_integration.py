#!/usr/bin/env python3
"""
Orchestrator Validation Integration
=====================================
Add this to ssot_generation_orchestrator.py to close the E2E loop.

After execute() returns with a prompt_id, call validate_and_record()
to verify actual output and write results back to the SSOT.

Usage:
    orch = SSOTOrchestrator()
    plan = orch.plan_generation("Generate Kai fighting goblins")
    result = orch.execute(plan)
    verdict = orch.validate_and_record(result, plan)

The complete E2E flow becomes:
    1. plan_generation()   → Content analysis + SSOT lookup + resource selection
    2. execute()           → Submit to ComfyUI + wait for completion
    3. validate_and_record() → Verify output + quality gates + write to SSOT
"""

# --- Add this import at the top of ssot_generation_orchestrator.py ---
# from generation_output_validator import OutputValidator, print_verdict


# --- Add this method to the SSOTOrchestrator class ---

class SSOTOrchestratorValidationMixin:
    """
    Mixin that adds output validation to SSOTOrchestrator.
    Copy validate_and_record() into your SSOTOrchestrator class.
    """

    def validate_and_record(self, execute_result: dict,
                            plan: "GenerationPlan" = None) -> dict:
        """
        Validate generation output and record to SSOT.

        Args:
            execute_result: Return value from self.execute(plan)
            plan: The generation plan (for metadata)

        Returns:
            Dict with validation verdict and SSOT recording status
        """
        # Import here to avoid circular deps if files are separate
        try:
            from generation_output_validator import (
                OutputValidator, print_verdict
            )
        except ImportError:
            # Validator not installed — return basic result
            logger.warning(
                "generation_output_validator not found. "
                "Install at /opt/tower-echo-brain/scripts/"
            )
            return {
                "validation": "skipped",
                "reason": "validator not installed",
                "execute_result": execute_result,
            }

        prompt_id = execute_result.get("prompt_id")
        if not prompt_id:
            return {
                "validation": "skipped",
                "reason": f"No prompt_id in result: {execute_result.get('error', '')}",
            }

        if execute_result.get("error"):
            return {
                "validation": "failed",
                "reason": execute_result["error"],
            }

        # Run validation
        validator = OutputValidator()
        verdict = validator.validate_generation(
            prompt_id, record_ssot=True
        )

        # Enrich verdict with plan metadata
        if plan and plan.resources:
            verdict.model_used = verdict.model_used or plan.resources.checkpoint
            verdict.prompt_text = verdict.prompt_text or plan.resources.positive_prompt

        if plan and plan.fresh_data:
            # Tag which SSOT records were used
            ssot_sources = [
                f"{rec.table}.{rec.id}" for rec in plan.fresh_data
            ]
        else:
            ssot_sources = []

        # Print verdict
        print_verdict(verdict)

        return {
            "validation": verdict.status,
            "total_images": verdict.total_images,
            "passed_images": verdict.passed_images,
            "failed_images": verdict.failed_images,
            "issues": verdict.issues,
            "ssot_recorded": verdict.recorded_to_ssot,
            "ssot_sources": ssot_sources,
            "prompt_id": prompt_id,
            "images": [
                {
                    "path": img.path,
                    "passed": img.passed,
                    "dimensions": f"{img.width}x{img.height}",
                    "size_kb": round(img.file_size_bytes / 1024, 1),
                    "issues": img.issues,
                }
                for img in verdict.images
            ],
        }


# --- Updated CLI entry point for ssot_generation_orchestrator.py ---
# Replace the if __name__ == "__main__" block with this:

UPDATED_CLI = '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSOT-Compliant Generation Orchestrator"
    )
    parser.add_argument("prompt", nargs="?",
                        default="Generate Kai fighting cyberpunk goblins in Tokyo")
    parser.add_argument("--plan-only", action="store_true",
                        help="Show plan without submitting to ComfyUI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Alias for --plan-only")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip output validation")

    args = parser.parse_args()

    orch = SSOTOrchestrator()
    plan = orch.plan_generation(args.prompt)
    print_plan(plan)

    if args.plan_only or args.dry_run:
        print("\\n  [--plan-only] Skipping ComfyUI submission\\n")
    else:
        print("\\n  Submitting to ComfyUI...")
        result = orch.execute(plan)
        print(f"\\n  Execute result: {json.dumps(result, indent=2)}")

        if not args.no_validate and result.get("prompt_id"):
            print("\\n  Validating output...")
            validation = orch.validate_and_record(result, plan)
            print(f"\\n  Validation: {json.dumps(validation, indent=2, default=str)}")
'''
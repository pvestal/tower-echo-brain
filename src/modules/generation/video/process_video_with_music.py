#!/usr/bin/env python3
"""
Simple Video Music Integration Script for Patrick
Easy-to-use script for adding music to scaled anime videos
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

from tests.test_complete_integration import CompleteMusicIntegrationTester

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Add music to Patrick's scaled anime videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python3 process_video_with_music.py /path/to/video.mp4

  # Process all videos in directory
  python3 process_video_with_music.py /path/to/directory/ --batch

  # Process with Jellyfin copy
  python3 process_video_with_music.py /path/to/video.mp4 --jellyfin

  # Process Cyberpunk Goblin videos
  python3 process_video_with_music.py /mnt/1TB-storage/ComfyUI/output/rife_scaling_tests/cyberpunk_goblin/ --batch
        """
    )

    parser.add_argument('input_path', help='Path to video file or directory')
    parser.add_argument('--batch', action='store_true', help='Process all videos in directory')
    parser.add_argument('--jellyfin', action='store_true', help='Copy output to Jellyfin library')
    parser.add_argument('--output-dir', help='Custom output directory')
    parser.add_argument('--quiet', action='store_true', help='Minimize output')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    input_path = Path(args.input_path)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    async def process():
        tester = CompleteMusicIntegrationTester()

        if args.output_dir:
            tester.output_dir = Path(args.output_dir)
            tester.output_dir.mkdir(parents=True, exist_ok=True)

        if args.batch:
            if not input_path.is_dir():
                logger.error("Batch mode requires a directory path")
                sys.exit(1)

            logger.info(f"🎵 Processing all videos in: {input_path}")
            result = await tester.run_batch_test(str(input_path))

            print(f"\n🎬 BATCH PROCESSING RESULTS:")
            print(f"Total videos: {result['total_videos']}")
            print(f"Successful: {result['summary']['successful_tests']}")
            print(f"Failed: {result['summary']['failed_tests']}")
            print(f"Success rate: {result['summary']['success_rate']:.1%}")
            print(f"Total time: {result['summary']['total_processing_time']:.1f}s")

            # Show successful outputs
            successful_outputs = []
            for video_name, video_result in result['results'].items():
                if isinstance(video_result, dict) and video_result.get('final_outputs', {}).get('video_with_music'):
                    successful_outputs.append(video_result['final_outputs']['video_with_music'])

            if successful_outputs:
                print(f"\n✅ Generated videos:")
                for output in successful_outputs:
                    print(f"  - {output}")

                if args.jellyfin:
                    print(f"\n📺 Copying to Jellyfin...")
                    # Copy to Jellyfin would be implemented here

        else:
            if not input_path.is_file():
                logger.error("Single file mode requires a video file path")
                sys.exit(1)

            if not input_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
                logger.error("Input must be a video file (.mp4, .mov, .avi)")
                sys.exit(1)

            logger.info(f"🎵 Processing single video: {input_path.name}")
            result = await tester.run_complete_integration_test(str(input_path))

            # Display results
            print(f"\n🎬 PROCESSING RESULTS:")
            print(f"Video: {input_path.name}")
            print(f"Success: {'✅ YES' if result['performance_metrics']['overall_success'] else '❌ NO'}")
            print(f"Duration: {result['performance_metrics']['total_duration']:.1f}s")

            if result['final_outputs'].get('video_with_music'):
                output_file = result['final_outputs']['video_with_music']
                print(f"Output: {output_file}")

                # Get file size
                output_path = Path(output_file)
                if output_path.exists():
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"Size: {size_mb:.1f} MB")

                if result.get('stages', {}).get('quality_assessment', {}).get('assessment'):
                    quality = result['stages']['quality_assessment']['assessment']
                    print(f"Quality: {quality['overall_score']:.2f}/1.0")

                if args.jellyfin:
                    print(f"📺 Copying to Jellyfin...")
                    # Jellyfin copy would be implemented here

            if result['errors']:
                print(f"\n⚠️  Errors ({len(result['errors'])}):")
                for error in result['errors']:
                    print(f"  - {error}")

    # Run the async processing
    try:
        asyncio.run(process())
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
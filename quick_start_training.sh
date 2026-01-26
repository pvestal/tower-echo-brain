#!/bin/bash
# Quick Start Training Script - User Journey Integration

set -e

echo "🎬 TOWER LORA TRAINING STUDIO - QUICK START"
echo "=========================================="
echo

# Show current status
echo "📊 Current Training Status:"
/opt/tower-lora-studio/train_manager.sh status | head -30
echo

# Menu for user
echo "🎯 Training Options:"
echo "1. Train a specific LoRA (recommended for testing)"
echo "2. Queue all NSFW LoRAs for training"
echo "3. Queue all violence/gore LoRAs"
echo "4. Start continuous training worker"
echo "5. View detailed training queue"
echo "6. Clear failed jobs and restart"
echo

read -p "Choose option (1-6): " choice

case $choice in
    1)
        echo
        echo "🔥 Popular LoRAs to train:"
        echo "   cowgirl_position    - NSFW intimate scene"
        echo "   blood_splatter     - Violence/gore effect"
        echo "   sword_slash        - Action combat"
        echo "   cyberpunk          - Visual style"
        echo "   bedroom            - Scene setting"
        echo
        read -p "Enter LoRA name: " lora_name

        if [ ! -z "$lora_name" ]; then
            echo
            echo "🚀 Starting training for: $lora_name"
            echo "This will:"
            echo "  1. Collect videos from Tower anime productions"
            echo "  2. Extract and enhance training frames"
            echo "  3. Train LoRA using diffusers with proper settings"
            echo "  4. Save trained model to /mnt/1TB-storage/models/loras/"
            echo "  5. Update database and integrate with ComfyUI"
            echo
            read -p "Continue? (y/n): " confirm

            if [ "$confirm" = "y" ]; then
                echo
                echo "⏳ Training started..."
                /opt/tower-lora-studio/train_manager.sh train-specific "$lora_name"

                echo
                echo "✅ Training complete! Check results:"
                echo "   Models: /mnt/1TB-storage/models/loras/${lora_name}*.safetensors"
                echo "   Logs: /opt/tower-lora-studio/logs/"
                echo "   Status: /opt/tower-lora-studio/train_manager.sh status"
            fi
        fi
        ;;

    2)
        echo
        echo "🔞 Queueing all NSFW LoRAs for training..."
        /opt/tower-lora-studio/train_manager.sh queue-all-nsfw

        echo
        echo "🤖 Start training worker? This will run continuously."
        read -p "Start worker? (y/n): " start_worker

        if [ "$start_worker" = "y" ]; then
            echo "🔄 Starting training worker..."
            /opt/tower-lora-studio/train_manager.sh train-worker
        fi
        ;;

    3)
        echo
        echo "🗡️ Queueing all violence/gore LoRAs for training..."
        /opt/tower-lora-studio/train_manager.sh queue-violence

        echo
        echo "🤖 Start training worker? This will run continuously."
        read -p "Start worker? (y/n): " start_worker

        if [ "$start_worker" = "y" ]; then
            echo "🔄 Starting training worker..."
            /opt/tower-lora-studio/train_manager.sh train-worker
        fi
        ;;

    4)
        echo
        echo "🔄 Starting continuous training worker..."
        echo "This will process all queued LoRAs automatically."
        echo "Use Ctrl+C to stop."
        echo
        /opt/tower-lora-studio/train_manager.sh train-worker
        ;;

    5)
        echo
        /opt/tower-lora-studio/train_manager.sh list-queue
        ;;

    6)
        echo
        echo "🧹 Clearing failed jobs..."
        PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -c "
            UPDATE training_queue SET status = 'pending' WHERE status = 'failed';
            UPDATE lora_training_status SET status = 'not_started' WHERE status = 'failed';
        "
        echo "✅ Failed jobs reset to pending"
        ;;

    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo
echo "📋 Quick Commands:"
echo "  Status:     /opt/tower-lora-studio/train_manager.sh status"
echo "  Queue:      /opt/tower-lora-studio/train_manager.sh list-queue"
echo "  Violence:   /opt/tower-lora-studio/train_manager.sh violence-status"
echo "  Train one:  /opt/tower-lora-studio/train_manager.sh train-specific <name>"
echo "  Worker:     /opt/tower-lora-studio/train_manager.sh train-worker"
echo
echo "💡 Integration with anime production:"
echo "  - Trained LoRAs auto-appear in ComfyUI"
echo "  - Database tracks all training status"
echo "  - Videos from Tower productions used as training data"
echo "  - Models saved to centralized /mnt/1TB-storage/models/loras/"
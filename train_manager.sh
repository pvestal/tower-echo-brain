#!/bin/bash
# LoRA Training Manager Script
# Start training for specific LoRA types

set -e

ACTION=${1:-status}
LORA_NAME=$2
PRIORITY=${3:-5}

case "$ACTION" in
    status)
        echo "🎬 LoRA Training Status"
        python3 /opt/tower-lora-studio/src/lora_status_manager.py status
        ;;

    queue)
        if [ -z "$LORA_NAME" ]; then
            echo "Usage: $0 queue <lora_name> [priority]"
            exit 1
        fi
        python3 /opt/tower-lora-studio/src/lora_status_manager.py queue --lora "$LORA_NAME" --priority $PRIORITY
        ;;

    queue-all-nsfw)
        echo "Adding all NSFW LoRAs to training queue..."
        PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -t -c "
            SELECT name FROM lora_definitions WHERE is_nsfw = true
        " | while read lora; do
            if [ ! -z "$lora" ]; then
                python3 /opt/tower-lora-studio/src/lora_status_manager.py queue --lora "$lora" --priority 10
            fi
        done
        ;;

    queue-all-poses)
        echo "Adding all pose LoRAs to training queue..."
        PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -t -c "
            SELECT name FROM lora_definitions WHERE category_id = (SELECT id FROM lora_categories WHERE name='pose')
        " | while read lora; do
            if [ ! -z "$lora" ]; then
                python3 /opt/tower-lora-studio/src/lora_status_manager.py queue --lora "$lora" --priority 8
            fi
        done
        ;;

    queue-violence)
        echo "Adding all violence/gore LoRAs to training queue..."
        PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -t -c "
            SELECT name FROM lora_definitions d
            JOIN lora_categories c ON d.category_id = c.id
            WHERE c.name IN ('violence', 'gore', 'weapons', 'death')
               OR d.name LIKE '%goblin_slayer%'
        " | while read lora; do
            if [ ! -z "$lora" ]; then
                python3 /opt/tower-lora-studio/src/lora_status_manager.py queue --lora "$lora" --priority 10
            fi
        done
        ;;

    violence-status)
        echo "🗡️ Violence/Gore LoRA Status"
        PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -c "
            SELECT status_icon, category, display_name, trigger_word, training_status
            FROM violence_gore_loras
            ORDER BY category, lora_name;
        "
        ;;

    train-next)
        echo "Processing next item in training queue..."
        python3 /opt/tower-lora-studio/src/tower_ltx_trainer.py
        ;;

    train-worker)
        echo "Starting REAL diffusers training worker (continuous)..."
        while true; do
            python3 /opt/tower-lora-studio/src/tower_ltx_trainer.py
            echo "Waiting 60 seconds before checking queue again..."
            sleep 60
        done
        ;;

    build-dataset)
        if [ -z "$LORA_NAME" ]; then
            echo "Usage: $0 build-dataset <lora_name>"
            exit 1
        fi
        echo "Building training dataset for $LORA_NAME..."
        python3 /opt/tower-lora-studio/src/dataset_manager.py "$LORA_NAME"
        ;;

    train-specific)
        if [ -z "$LORA_NAME" ]; then
            echo "Usage: $0 train-specific <lora_name>"
            exit 1
        fi
        echo "Training specific LoRA: $LORA_NAME"

        # First queue it with high priority
        python3 /opt/tower-lora-studio/src/lora_status_manager.py queue --lora "$LORA_NAME" --priority 10

        # Then process immediately
        python3 /opt/tower-lora-studio/src/tower_ltx_trainer.py
        ;;

    list-queue)
        echo "📋 Training Queue:"
        PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -c "
            SELECT
                q.priority,
                d.name as lora_name,
                d.trigger_word,
                q.status,
                q.scheduled_at
            FROM training_queue q
            JOIN lora_definitions d ON q.definition_id = d.id
            WHERE q.status IN ('pending', 'processing')
            ORDER BY q.priority DESC, q.scheduled_at
        "
        ;;

    clear-queue)
        echo "Clearing pending items from queue..."
        PGPASSWORD=RP78eIrW7cI2jYvL5akt1yurE psql -h localhost -U patrick -d anime_production -c "
            DELETE FROM training_queue WHERE status = 'pending'
        "
        ;;

    *)
        echo "Usage: $0 {status|queue|queue-all-nsfw|queue-all-poses|train-next|train-worker|list-queue|clear-queue}"
        echo ""
        echo "Commands:"
        echo "  status         - Show training status of all LoRAs"
        echo "  queue NAME     - Add specific LoRA to training queue"
        echo "  queue-all-nsfw - Add all NSFW LoRAs to queue"
        echo "  queue-all-poses - Add all pose LoRAs to queue"
        echo "  train-next     - Train next item in queue"
        echo "  train-worker   - Start continuous training worker"
        echo "  list-queue     - Show current training queue"
        echo "  clear-queue    - Clear pending queue items"
        exit 1
        ;;
esac
-- Violence, Gore, and Action LoRAs for Cyberpunk/Goblin Slayer content
-- Adult content for mature anime production

-- Add new categories for violence content
INSERT INTO lora_categories (name, description, priority) VALUES
('violence', 'Violence and combat actions', 10),
('gore', 'Blood and gore effects', 9),
('weapons', 'Weapons and combat equipment', 8),
('death', 'Death and injury states', 7)
ON CONFLICT (name) DO NOTHING;

-- Violence and Combat LoRAs
INSERT INTO lora_definitions (category_id, name, display_name, trigger_word, description, base_prompt, is_nsfw, training_priority)
VALUES
-- Violence Actions
((SELECT id FROM lora_categories WHERE name='violence'), 'sword_slash', 'Sword Slash Attack', 'sword_slash', 'Sword slashing motion with impact', 'sword slashing through target, blade motion, impact effect', true, 10),
((SELECT id FROM lora_categories WHERE name='violence'), 'decapitation', 'Decapitation', 'decapitation', 'Beheading action', 'decapitation, head being severed, violent motion', true, 10),
((SELECT id FROM lora_categories WHERE name='violence'), 'stabbing', 'Stabbing Attack', 'stabbing', 'Stabbing penetration motion', 'stabbing motion, blade penetrating body', true, 9),
((SELECT id FROM lora_categories WHERE name='violence'), 'dismemberment', 'Dismemberment', 'dismemberment', 'Limb removal', 'dismemberment, limb being severed, violent', true, 10),
((SELECT id FROM lora_categories WHERE name='violence'), 'neck_snap', 'Neck Snap', 'neck_snap', 'Breaking neck motion', 'neck snapping, violent twist', true, 8),
((SELECT id FROM lora_categories WHERE name='violence'), 'brutal_beating', 'Brutal Beating', 'brutal_beating', 'Violent physical assault', 'brutal beating, punching, kicking, violence', true, 9),
((SELECT id FROM lora_categories WHERE name='violence'), 'torture', 'Torture Scene', 'torture', 'Torture and interrogation', 'torture scene, inflicting pain', true, 8),
((SELECT id FROM lora_categories WHERE name='violence'), 'execution', 'Execution', 'execution', 'Execution scene', 'execution, killing blow, final strike', true, 9),

-- Gore Effects
((SELECT id FROM lora_categories WHERE name='gore'), 'blood_splatter', 'Blood Splatter', 'blood_splatter', 'Blood spray effect', 'blood splattering, arterial spray, red liquid', true, 10),
((SELECT id FROM lora_categories WHERE name='gore'), 'blood_pool', 'Blood Pool', 'blood_pool', 'Pooling blood on ground', 'pool of blood, bleeding out, red puddle', true, 8),
((SELECT id FROM lora_categories WHERE name='gore'), 'gore_explosion', 'Gore Explosion', 'gore_explosion', 'Body parts exploding', 'gore explosion, body parts flying, visceral', true, 10),
((SELECT id FROM lora_categories WHERE name='gore'), 'entrails', 'Exposed Entrails', 'entrails', 'Internal organs visible', 'entrails, guts spilling out, viscera', true, 9),
((SELECT id FROM lora_categories WHERE name='gore'), 'severed_limbs', 'Severed Limbs', 'severed_limbs', 'Detached body parts', 'severed limbs, amputated arms and legs', true, 9),
((SELECT id FROM lora_categories WHERE name='gore'), 'crushed_skull', 'Crushed Skull', 'crushed_skull', 'Head crushing', 'skull being crushed, head trauma, brain matter', true, 10),
((SELECT id FROM lora_categories WHERE name='gore'), 'impalement', 'Impalement', 'impalement', 'Object through body', 'impalement, spike through body, pierced', true, 9),

-- Weapons
((SELECT id FROM lora_categories WHERE name='weapons'), 'katana', 'Katana Sword', 'katana', 'Japanese sword', 'katana blade, japanese sword, sharp edge', false, 7),
((SELECT id FROM lora_categories WHERE name='weapons'), 'battle_axe', 'Battle Axe', 'battle_axe', 'Large combat axe', 'battle axe, heavy weapon, double blade', false, 6),
((SELECT id FROM lora_categories WHERE name='weapons'), 'energy_blade', 'Energy Blade', 'energy_blade', 'Cyberpunk energy weapon', 'energy blade, plasma sword, glowing weapon, cyberpunk', false, 8),
((SELECT id FROM lora_categories WHERE name='weapons'), 'chainsaw', 'Chainsaw', 'chainsaw', 'Chainsaw weapon', 'chainsaw, rotating chain, mechanical weapon', true, 9),
((SELECT id FROM lora_categories WHERE name='weapons'), 'mace', 'Spiked Mace', 'spiked_mace', 'Medieval crushing weapon', 'spiked mace, crushing weapon, medieval', false, 6),

-- Death States
((SELECT id FROM lora_categories WHERE name='death'), 'death_throes', 'Death Throes', 'death_throes', 'Dying convulsions', 'death throes, dying, final moments', true, 8),
((SELECT id FROM lora_categories WHERE name='death'), 'corpse_pile', 'Corpse Pile', 'corpse_pile', 'Multiple dead bodies', 'pile of corpses, dead bodies stacked', true, 7),
((SELECT id FROM lora_categories WHERE name='death'), 'dying_breath', 'Dying Breath', 'dying_breath', 'Last breath', 'dying breath, final exhale, death rattle', true, 7),

-- Goblin Slayer Specific
((SELECT id FROM lora_categories WHERE name='character'), 'goblin_slayer_armor', 'Goblin Slayer Armor', 'goblin_slayer_armor', 'Iconic armor design', 'goblin slayer armor, metal plates, cyberpunk style, glowing visor', false, 10),
((SELECT id FROM lora_categories WHERE name='violence'), 'goblin_massacre', 'Goblin Massacre', 'goblin_massacre', 'Mass goblin killing', 'massacring goblins, multiple kills, brutal combat', true, 10),
((SELECT id FROM lora_categories WHERE name='action'), 'goblin_slayer_finisher', 'Goblin Slayer Finisher', 'goblin_slayer_finisher', 'Signature killing move', 'goblin slayer finishing move, brutal kill, signature attack', true, 10),

-- Cyberpunk Violence
((SELECT id FROM lora_categories WHERE name='violence'), 'cyber_augment_rip', 'Cyber Augment Ripping', 'cyber_augment_rip', 'Tearing out cybernetic parts', 'ripping out cybernetic augments, sparks, mechanical gore', true, 9),
((SELECT id FROM lora_categories WHERE name='gore'), 'neon_blood', 'Neon Blood Effect', 'neon_blood', 'Cyberpunk style glowing blood', 'neon glowing blood, cyberpunk gore, luminescent red', true, 8),
((SELECT id FROM lora_categories WHERE name='violence'), 'hack_and_slash', 'Hack and Slash', 'hack_and_slash', 'Rapid blade combat', 'hack and slash combat, multiple cuts, fast blade work', true, 9)
ON CONFLICT (name) DO NOTHING;

-- Create specialized view for violence/gore content
CREATE OR REPLACE VIEW violence_gore_loras AS
SELECT
    c.name as category,
    d.name as lora_name,
    d.display_name,
    d.trigger_word,
    d.description,
    COALESCE(s.status, 'not_started') as training_status,
    s.model_path,
    CASE
        WHEN s.model_path IS NOT NULL THEN '✅'
        WHEN s.status = 'training' THEN '🔄'
        WHEN s.status = 'queued' THEN '⏳'
        ELSE '❌'
    END as status_icon
FROM lora_definitions d
JOIN lora_categories c ON d.category_id = c.id
LEFT JOIN lora_training_status s ON d.id = s.definition_id
WHERE c.name IN ('violence', 'gore', 'weapons', 'death')
   OR d.name LIKE '%goblin_slayer%'
   OR d.name LIKE '%cyber%'
ORDER BY d.training_priority DESC, c.priority DESC, d.name;

-- Stats function for violence content
CREATE OR REPLACE FUNCTION get_violence_stats()
RETURNS TABLE(
    total_violence_loras INTEGER,
    trained INTEGER,
    violence_category INTEGER,
    gore_category INTEGER,
    weapons_category INTEGER,
    goblin_slayer_specific INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_violence_loras,
        COUNT(CASE WHEN s.status = 'completed' THEN 1 END)::INTEGER as trained,
        COUNT(CASE WHEN c.name = 'violence' THEN 1 END)::INTEGER as violence_category,
        COUNT(CASE WHEN c.name = 'gore' THEN 1 END)::INTEGER as gore_category,
        COUNT(CASE WHEN c.name = 'weapons' THEN 1 END)::INTEGER as weapons_category,
        COUNT(CASE WHEN d.name LIKE '%goblin_slayer%' THEN 1 END)::INTEGER as goblin_slayer_specific
    FROM lora_definitions d
    JOIN lora_categories c ON d.category_id = c.id
    LEFT JOIN lora_training_status s ON d.id = s.definition_id
    WHERE c.name IN ('violence', 'gore', 'weapons', 'death')
       OR d.name LIKE '%goblin_slayer%';
END;
$$ LANGUAGE plpgsql;
echo -900 > /proc/    $$/oom_score_adj 2>/dev/null

TASKS=(
    # "adjust_bottle"
    # "beat_block_hammer"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"

    "hanging_mug"
    "lift_pot"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "open_laptop"
    "open_microwave"
    "pick_diverse_bottles"
    "pick_dual_bottles"

    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_burger_fries"

    "place_can_basket"
    "place_cans_plasticbox"
    "place_container_plate"
    "place_dual_shoes"
    "place_empty_cup"

    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"

    "rotate_qrcode"
    "scan_object"
    "shake_bottle"
    "shake_bottle_horizontally"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
    "stamp_seal"
    "turn_switch"
)

export XDG_CACHE_HOME=/scratch/peilab/wzj/

BASE_DIR="/project/peilab/wzj/RoboTwin/policy/openpi_test/training_data/robotwin_50_task_clean"

FAILED_FILE="./FFFFFailed_tasks.txt"         # 日志保存目录，可自行修改路径
# mkdir -p "$LOG_DIR"

for task in "${TASKS[@]}"; do
    task_dir="${BASE_DIR}/${task}"
    
    echo "======================================"
    echo "正在处理任务: ${task}"
    echo "目录: ${task_dir}"
    
    if [ ! -d "${task_dir}" ]; then
        echo "  警告：目录 ${task_dir} 不存在，跳过"
        continue
    fi
    
    bash generate.sh "${task_dir}" "${task}" 
    
    ret=$?
    
    if [ $ret -eq 0 ]; then
        echo "  → 成功"
    else
        # 把失败的任务名追加写入文件
        echo "${task}" >> "$FAILED_FILE"
        echo "    已记录失败任务到 ${FAILED_FILE}"
    fi
    
    echo ""
done

echo "所有任务处理完毕。"

# uv run python3 scripts/merge_lerobot_subrepos_v2.py \
#   --input-repos /scratch/peilab/wzj/huggingface/lerobot/* \
#   --output-repo /scratch/peilab/wzj/.cache/huggingface/lerobot/robotwin_aloha_lerobot_repo \
#   --chunk_size 1000

# export XDG_CACHE_HOME=/scratch/peilab/wzj/.cache
# uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_full_torch

# sbatch train_torch_robotwin.sbatch

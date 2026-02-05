#!/bin/bash

# 批量评估脚本：测试 pi0_base_aloha_robotwin_full_torch checkpoint 在 50 个任务上的成功率
# 使用 repo: /project/peilab/wzj/.cache/huggingface/lerobot/robotwin_aloha_lerobot_repo
# 每个任务运行 100 次测试（基于 eval_policy.py 的 test_num=100）
# Checkpoint: 110000 (请根据实际 checkpoint step 修改)

TASK=(
    "adjust_bottle"
    "beat_block_hammer"
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

# 日志目录
EVAL_LOG_DIR="/project/peilab/wzj/RoboTwin/policy/openpi_test/eval_50_tasks_logs"
mkdir -p $EVAL_LOG_DIR

# 切换到 openpi_test 目录
cd /project/peilab/wzj/RoboTwin/policy/openpi_test

SUCCESS_RATES=()
TOTAL_SUCCESS_RATE=0
TOTAL_TASKS=${#TASK[@]}

for task in "${TASK[@]}"; do
    echo "======================================"
    echo "正在评估任务: ${task}"
    
    # 运行评估，同时输出到终端和日志文件
    bash eval.sh ${task} demo_clean pi0_base_aloha_robotwin_full_torch robotwin_aloha_lerobot 0 0 | tee $EVAL_LOG_DIR/${task}_eval.log
    
    # 从日志中提取成功率（格式如 "Success rate: X/Y => Z%"）
    SUCCESS_RATE=$(grep "Success rate:" $EVAL_LOG_DIR/${task}_eval.log | tail -1 | grep -oP '\d+\.\d+(?=%)' || echo "0.0")
    
    if [[ -z "$SUCCESS_RATE" ]]; then
        SUCCESS_RATE="0.0"
    fi
    
    SUCCESS_RATES+=("$SUCCESS_RATE")
    TOTAL_SUCCESS_RATE=$(echo "$TOTAL_SUCCESS_RATE + $SUCCESS_RATE" | bc -l)
    
    echo "  → 成功率: ${SUCCESS_RATE}%"
    echo ""
done

AVERAGE_SUCCESS_RATE=$(echo "scale=2; $TOTAL_SUCCESS_RATE / $TOTAL_TASKS" | bc -l)

echo "评估完成。"
echo "总任务数: $TOTAL_TASKS"
echo "每个任务的成功率:"
for i in "${!TASK[@]}"; do
    echo "  ${TASK[$i]}: ${SUCCESS_RATES[$i]}%"
done
echo "平均成功率: ${AVERAGE_SUCCESS_RATE}%"

# 保存结果到 JSON 文件
OUTPUT_JSON="/project/peilab/wzj/RoboTwin/policy/openpi_test/eval_50_tasks_results.json"
{
    echo "{"
    echo "  \"total_tasks\": $TOTAL_TASKS,"
    echo "  \"average_success_rate\": \"${AVERAGE_SUCCESS_RATE}%\","
    echo "  \"task_success_rates\": {"
    for i in "${!TASK[@]}"; do
        if [ $i -lt $((${#TASK[@]} - 1)) ]; then
            echo "    \"${TASK[$i]}\": \"${SUCCESS_RATES[$i]}%\","
        else
            echo "    \"${TASK[$i]}\": \"${SUCCESS_RATES[$i]}%\""
        fi
    done
    echo "  }"
    echo "}"
} > $OUTPUT_JSON

echo "结果已保存到 $OUTPUT_JSON"
torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_json_new.py \
  --test_json /scratch3/yan204/yxp/VideoX_Fun/data/test_json/4tasks_rem_add_swap_local-style_test.json \
  --output_dir lucy_edit_results\
  --seed 0 \
  --num_frames 33 \
  --source_frames 33 \
  --guidance_scale 5.0 \
  --negative_prompt  "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
  --model_id /scratch3/yan204/models/Lucy-Edit-Dev
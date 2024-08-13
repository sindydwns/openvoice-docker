import facade

speaker_path = "resources/sample.mp3"
speaker_se_path = "voice/sample.pth"
base_speaker_se_path = "checkpoints_v2/base_speakers/ses/kr.pth"
text = "안녕하세요! 오늘은 날씨가 정말 좋네요."
src_output_path = "outputs_v2/src_output.wav"
target_output_path = "outputs_v2/target_output.wav"

tone_color_converter = facade.create_tone_color_converter("checkpoints_v2")
model = facade.Model(tone_color_converter)
model.train(speaker_path, speaker_se_path)
model.load(base_speaker_se_path, speaker_se_path)
model.tts(text, src_output_path)
model.tone_color(src_output_path, target_output_path)

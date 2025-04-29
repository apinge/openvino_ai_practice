import outetts

# Configure the model
model_config = outetts.HFModelConfig_v2(
    model_path="OuteAI/OuteTTS-0.3-1B",
    tokenizer_path="OuteAI/OuteTTS-0.3-1B"
)
# Initialize the interface
interface = outetts.InterfaceHF(model_version="0.3", cfg=model_config)

# You can create a speaker profile for voice cloning, which is compatible across all backends.
# speaker = interface.create_speaker(audio_path="path/to/audio/file.wav")
# interface.save_speaker(speaker, "speaker.json")
# speaker = interface.load_speaker("speaker.json")

# Print available default speakers
interface.print_default_speakers()
# Load a default speaker
"""
Available default speakers v2: ['de_male_1', 'de_male_2', 'en_female_1', 'en_male_1', 'en_male_2', 'en_male_3', 
'fr_female_1', 'fr_female_2', 'fr_male_1', 'fr_male_2', 'jp_female_1', 'jp_female_2', 'jp_male_1', 'ko_female_1', 
'zh_female_1', 'zh_female_2', 'zh_male_1', 'zh_male_2']
"""
speaker = interface.load_default_speaker(name="zh_male_2")

# Generate speech
gen_cfg = outetts.GenerationConfig(
    text="开源社区的代码架构比较复杂.",
    temperature=0.4,
    repetition_penalty=1.1,
    max_length=4096,
    speaker=speaker,
)
output = interface.generate(config=gen_cfg)

# Save the generated speech to a file
output.save("output_torch.wav")
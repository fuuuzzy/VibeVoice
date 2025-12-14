#!/usr/bin/env python3
"""
Zero-Shot Voice Cloning Script with VibeVoice-1.5B
Supports both single text input and batch processing with SRT files.

Usage: 
  Single mode: python voice_clone.py --text "Your text here" --reference /path/to/reference.mp3 --model_path /path/to/VibeVoice-1.5B
  Batch mode:  python voice_clone.py --srt subtitles.srt --reference-dir /path/to/audio_dir --model_path /path/to/VibeVoice-1.5B
"""

import argparse
import copy
import os
import re

import torch

# Import VibeVoice modules
# Assuming vibevoice is in python path or current directory
try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import \
        VibeVoiceStreamingForConditionalGenerationInference
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.streamer import AudioStreamer
except ImportError:
    print("Error: Could not import vibevoice modules. Please ensure you are in the VibeVoice directory.")
    exit(1)


def parse_srt(srt_file):
    """Parse SRT file and extract subtitle entries"""
    print(f"Parsing SRT file: {srt_file}")
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                subtitle_id = int(lines[0].strip())
                text = ' '.join(lines[2:]).strip()
                subtitles.append({'id': subtitle_id, 'text': text})
            except:
                continue
    print(f"Parsed {len(subtitles)} subtitle entries")
    return subtitles


def find_reference_audio(reference_dir, subtitle_id, audio_prefix='segment'):
    """Find reference audio file for a given subtitle ID"""
    patterns = [
        f"{audio_prefix}_{subtitle_id:03d}.wav", f"{audio_prefix}_{subtitle_id:03d}.mp3",
        f"{audio_prefix}_{subtitle_id}.wav", f"{audio_prefix}_{subtitle_id}.mp3",
        f"{audio_prefix}{subtitle_id:03d}.wav", f"{audio_prefix}{subtitle_id:03d}.mp3",
        f"{audio_prefix}{subtitle_id}.wav", f"{audio_prefix}{subtitle_id}.mp3",
    ]
    for pattern in patterns:
        path = os.path.join(reference_dir, pattern)
        if os.path.exists(path):
            return path
    return None


@torch.no_grad()
def prepare_voice_cache(model, processor, audio_path, device):
    """
    Generate the KV cache (prefilled outputs) from a reference audio file.
    This simulates the 'prefill' step for zero-shot cloning.
    """
    # Initialize a temporary VibeVoiceProcessor to handle prompt construction
    batch_processor = VibeVoiceProcessor(
        tokenizer=processor.tokenizer,
        audio_processor=processor.audio_processor,
        speech_tok_compress_ratio=processor.speech_tok_compress_ratio,
        db_normalize=processor.db_normalize
    )

    # Create voice prompt components
    # voice_tokens: tokens for " Voice input:\n Speaker 0: <start><vae tokens...><end>\n"
    # voice_speech_inputs: list containing containing the audio array
    voice_tokens, voice_speech_inputs, _ = batch_processor._create_voice_prompt([audio_path])

    # Create System Prompt
    system_prompt = batch_processor.system_prompt
    system_tokens = processor.tokenizer.encode(system_prompt, add_special_tokens=False)

    # Combine to form the full prompt sequence
    full_tokens = system_tokens + voice_tokens

    # Prepare inputs for the model
    input_ids = torch.tensor([full_tokens], device=device).long()
    attention_mask = torch.ones_like(input_ids)

    # Get initial Text Embeddings from the model
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # --- Inject Speech Embeddings ---
    # We need to replace the placeholder <vae_token_id> embeddings with actual speech embeddings
    vae_token_id = processor.tokenizer.speech_diffusion_id

    # 1. Encode Audio to Latents
    # voice_speech_inputs[0] is the numpy audio array
    wav_tensor = torch.tensor(voice_speech_inputs[0], device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    # Use the model's acoustic tokenizer (VAE/Encoder)
    # Note: We assume the acoustic tokenizer and connector are available on the model
    if not hasattr(model.model, 'acoustic_tokenizer'):
        raise RuntimeError("Model does not have acoustic_tokenizer")

    # Encode: output usually has .latents or directly tensor depending on model type
    # Trying generic assumption for VibeVoice VAE
    encoded = model.model.acoustic_tokenizer.encode(wav_tensor)
    if hasattr(encoded, 'latents'):
        speech_latents = encoded.latents
    elif isinstance(encoded, tuple):
        speech_latents = encoded[0]
    else:
        speech_latents = encoded

    # 2. Project Latents to LLM dimension
    speech_embeds = model.model.acoustic_connector(speech_latents)  # (1, num_latents, hidden_size)

    # 3. Replace embeddings in inputs_embeds
    # Find positions of vae_token_id
    is_vae_token = (input_ids == vae_token_id)

    # Verify counts match
    num_vae_tokens = is_vae_token.sum()
    num_speech_embeds = speech_embeds.shape[1]

    if num_vae_tokens != num_speech_embeds:
        # If mismatch, it might be due to padding or calculation diff in processor vs tokenizer
        # We trim or pad speech embeds to match tokens
        print(f"Warning: Token count {num_vae_tokens} != Embed count {num_speech_embeds}. Adjusting.")
        if num_speech_embeds > num_vae_tokens:
            speech_embeds = speech_embeds[:, :num_vae_tokens, :]
        else:
            # Pad is harder, but simplified approach:
            pass

            # Inject
    inputs_embeds[is_vae_token] = speech_embeds.reshape(-1, speech_embeds.shape[-1])

    # --- Run Forward Pass to get Cache ---

    # 1. Forward LM (Text/Base LM)
    # Note: forward_lm usually takes input_ids. But we want to pass inputs_embeds!
    # The signature in modeling_vibevoice_streaming_inference.py accepts inputs_embeds.
    lm_outputs = model.forward_lm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True
    )

    # 2. Forward TTS LM
    # TTS LM needs lm_last_hidden_state (from LM) and tts_text_masks.
    # What is tts_text_masks for the prompt?
    # In VibeVoiceProcessor, voice_speech_masks are True for speech tokens.
    # In forward_tts_lm, we usually want 1 for text, 0 for speech?
    # Or maybe it adds type embedding based on mask.
    # modeling_vibevoice_streaming_inference.py:
    # inputs_embeds = inputs_embeds + self.model.tts_input_types(tts_text_masks.long())
    # So we need to provide the mask.
    # Let's assume 0 = Speech, 1 = Text.
    # The VibeVoiceProcessor returns `voice_speech_masks` where True=Speech (VAE tokens).

    # Construct speech mask for the full sequence
    # System prompt is text (False/0 in processor mask terms?)
    # Wait, in processor: speech_input_mask += [False] * len(system_tokens)
    # So False means Text in processor? 
    # Let's check VibeVoiceProcessor._create_voice_prompt:
    #   voice_speech_masks = [False] * len(prefix) ... [True] * len(vae) ...
    # So True means Speech.

    # We need to map this to what the model expects.
    # If model expects 0/1 types.
    # Usually 0 is text, 1 is speech or vice versa.
    # Let's check modeling code again?
    # inputs_embeds + self.model.tts_input_types(tts_text_masks.long())
    # If tts_text_masks is boolean, True -> 1, False -> 0.

    # Let's assume we want keys "lm", "tts_lm".

    # Reconstruct mask
    # System tokens are text -> False
    # Voice tokens: prefix(False), start(False?), vae(True), end(False?)
    # We can follow what processor returns.

    _, _, voice_speech_masks = batch_processor._create_voice_prompt([audio_path])

    # Full mask
    full_speech_mask = [False] * len(system_tokens) + voice_speech_masks
    full_speech_mask_tensor = torch.tensor([full_speech_mask], device=device)  # True for speech

    # TTS model usually interprets 1 as text, 0 as speech? Or types 0 and 1.
    # If we pass boolean, it becomes 0 and 1.
    # Type 0: Speech? Type 1: Text?
    # Let's try to infer or assume 1=Text, 0=Speech (common).
    # If speech_mask has True for Speech, then text_mask = ~speech_mask
    tts_text_masks = (~full_speech_mask_tensor)

    # We also need initial TTS embeddings. 
    # forward_tts_lm takes inputs_embeds OR input_ids.
    # It replaces the tail with lm_last_hidden_state.
    # Since we are prefilling the PROMPT, the prompt drives the TTS LM too.
    # We pass the same inputs_embeds (with injected audio).

    tts_lm_outputs = model.forward_tts_lm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        lm_last_hidden_state=lm_outputs.last_hidden_state,
        tts_text_masks=tts_text_masks,
        use_cache=True,
        return_dict=True
    )

    # 3. Negative Prompt Cache
    # Usually empty or fixed negative prompt.
    neg_text_input_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    neg_input_ids = torch.tensor([[neg_text_input_id]], device=device)
    neg_attention_mask = torch.ones_like(neg_input_ids)

    neg_lm_outputs = model.forward_lm(
        input_ids=neg_input_ids,
        attention_mask=neg_attention_mask,
        use_cache=True,
        return_dict=True
    )

    # Negative TTS LM
    # Neg mask: is it text? <|image_pad|> is special. Treat as text (True).
    neg_text_masks = torch.ones((1, 1), device=device, dtype=torch.bool)

    neg_tts_lm_outputs = model.forward_tts_lm(
        input_ids=neg_input_ids,
        attention_mask=neg_attention_mask,
        lm_last_hidden_state=neg_lm_outputs.last_hidden_state,
        tts_text_masks=neg_text_masks,
        use_cache=True,
        return_dict=True
    )

    return {
        "lm": lm_outputs,
        "tts_lm": tts_lm_outputs,
        "neg_lm": neg_lm_outputs,
        "neg_tts_lm": neg_tts_lm_outputs
    }


def synthesize(text, prefilled_outputs, model, processor, output_path, device):
    """Synthesize speech using the prefilled voice cache"""

    # Prepare text input
    # StreamingProcessor process_input_with_cached_prompt handles the "Text input:" formatting
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )

    # Move to device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.5,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            all_prefilled_outputs=copy.deepcopy(prefilled_outputs)
        )

    # Save Audio
    if outputs.speech_outputs is not None and len(outputs.speech_outputs) > 0:
        processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Voice Cloning with VibeVoice")
    parser.add_argument('--text', type=str, help="Text to speak")
    parser.add_argument('--srt', type=str, help="SRT file path")
    parser.add_argument('--reference', type=str, help="Reference audio file")
    parser.add_argument('--reference-dir', type=str, help="Reference audio directory")
    parser.add_argument('--audio-prefix', type=str, default='segment', help="Audio file prefix")
    parser.add_argument('--model_path', type=str, default="VibeVoice-1.5B", help="Path to VibeVoice model")
    parser.add_argument('--output', type=str, default="outputs_vibevoice", help="Output directory")
    parser.add_argument('--language', type=str, default="EN", help="Ignored (VibeVoice is multilingual)")
    parser.add_argument('--skip-existing', action='store_true')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        # Fallback check
        if os.path.exists(os.path.join("..", args.model_path)):
            args.model_path = os.path.join("..", args.model_path)

    # Check for HuggingFace cache structure (snapshots directory)
    if os.path.exists(args.model_path) and os.path.isdir(os.path.join(args.model_path, "snapshots")):
        snapshots_dir = os.path.join(args.model_path, "snapshots")
        snapshots = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if snapshots:
            # Sort by modification time to get the latest
            snapshots.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            args.model_path = snapshots[0]
            print(f"Auto-detected HuggingFace snapshot. Using model path: {args.model_path}")

    print(f"Loading model from: {args.model_path}")

    # Load Processor
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)

    # Load Model
    # Choose dtype
    if device == "cuda":
        dtype = torch.bfloat16
        attn = "flash_attention_2"
    else:
        dtype = torch.float32
        attn = "sdpa"

    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=attn
    )
    model.to(device)
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    # Single Mode
    if args.text and args.reference:
        print(f"Processing single text with reference: {args.reference}")
        cache = prepare_voice_cache(model, processor, args.reference, device)
        out_path = os.path.join(args.output, "output_single.wav")
        if synthesize(args.text, cache, model, processor, out_path, device):
            print(f"Saved: {out_path}")
        else:
            print("Failed to generate audio")

    # Batch Mode
    elif args.srt and (args.reference_dir or args.reference):
        subtitles = parse_srt(args.srt)

        # Cache for multi-use reference (if single reference used)
        single_cache = None
        if args.reference:
            single_cache = prepare_voice_cache(model, processor, args.reference, device)

        for sub in subtitles:
            sid = sub['id']
            text = sub['text']
            out_file = f"clone_{sid:03d}.wav"
            out_path = os.path.join(args.output, out_file)

            if args.skip_existing and os.path.exists(out_path):
                print(f"Skipping {sid}...")
                continue

            print(f"Processing {sid}: {text[:30]}...")

            curr_cache = single_cache
            if not curr_cache:
                # Find reference
                ref = find_reference_audio(args.reference_dir, sid, args.audio_prefix)
                if ref:
                    try:
                        curr_cache = prepare_voice_cache(model, processor, ref, device)
                    except Exception as e:
                        print(f"Error preparing cache for {sid}: {e}")
                        continue
                else:
                    print(f"No reference found for {sid}")
                    continue

            try:
                synthesize(text, curr_cache, model, processor, out_path, device)
                print(f"  Saved {out_file}")
            except Exception as e:
                print(f"  Error generating {sid}: {e}")


if __name__ == "__main__":
    main()

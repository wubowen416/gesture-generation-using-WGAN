from google.cloud import speech
import os
import codecs
import json


def transcribe_file(audio_filename, text_filename):
    """Transcribe the given audio file."""

    client = speech.SpeechClient()

    with open(audio_filename, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='ja-JP',
        audio_channel_count=2,
        enable_word_time_offsets=True,
        # enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)
    if response.results == []:
        print(f"Recognization failed at {audio_filename}.")
    else:
        words = {
            'word_list': []
        }
        for i, sentence in enumerate(response.results):
            sentence = sentence.alternatives[0]
            print("Transcript: {}".format(sentence.transcript))
            # print("Confidence: {}".format(sentence.confidence))

            for word_info in sentence.words:
                word = word_info.word
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()

                # Append current word
                words['word_list'].append({
                    'word': word.split('|')[0],
                    'start_time': start_time,
                    'end_time': end_time
                })

        with codecs.open(text_filename, 'w', 'utf8') as f:
            f.write(json.dumps(words, ensure_ascii=False, indent=4))
            

if __name__ == '__main__':

    speech_dir = './data/takekuchi/source/speech'
    text_dir = './data/takekuchi/source/text'

    os.makedirs(text_dir, exist_ok=True)
    for speech_filename in os.listdir(speech_dir):
        text_filename = speech_filename.replace('audio', 'text').replace('.wav', '.json')
        transcribe_file(os.path.join(speech_dir, speech_filename), os.path.join(text_dir, text_filename))
#libraries to determine length of wav file
import wave
import contextlib

#library to load env variables
from dotenv import load_dotenv
import os
import io

#libraries to make video and process audio
from pydub import AudioSegment
from moviepy.editor import *
#import gizeh
import moviepy.editor as mpy
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip

#import Google Cloud client Library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

#syllable separator
import pyphen

#import audio library
import numpy as np, scipy as sp, sklearn, librosa, cmath, math, matplotlib.pyplot as plt
#from IPython.display import Audio

#import libraries to accept input from console
import sys

load_dotenv()

#set credentials
def loadCredentials():
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './LyricTracker-2b18d11f97c8.json'
#function that takes in a wav stereo and produces a mono version.
#could be replaced by api from wav converter site
#or manual change for now

def getFilePath(filename):
	return os.path.join(os.path.dirname(__file__), 'soundfiles', 'foreground', filename)	

#Function that turns multi channel to single channel, overwrites previous file
def stereoToMono(filepath):
	sound = AudioSegment.from_wav(filepath)
	sound = sound.set_channels(1)
	sound.export(filepath, format="wav")

#converts duration from google api to seconds
def durationToSec(duration):
	sec = duration.seconds
	nano = duration.nanos
	sec = float(sec) + float(nano)/(10.0**9)
	return sec

#makes start and end times to seconds
def convertToPython(wordInfo):
	startEnd = {}
#	print(wordInfo.word, wordInfo.start_time.seconds, wordInfo.start_time.nanos)
	startEnd["word"] = wordInfo.word
	startEnd["start"] = durationToSec(wordInfo.start_time)
	startEnd["end"] = durationToSec(wordInfo.end_time)
	return startEnd

#makes API Call
def getSpeechInfo(path2File):
	#gets path to file. Assumes its in soundfiles/foreground/ directory	
	client = speech.SpeechClient()
	
	#loading audio into proper file
	stereoToMono(path2File)
	with io.open(path2File, 'rb') as audio_file:
		content = audio_file.read()
		audio = types.RecognitionAudio(content = content)

	print('Waiting for transcription to complete...')
	# test different models and return the one with the highest connfidence
	highest_confidence = 0
	for model in ['default','command_and_search', 'phone_call', 'video']:
	
		config = types.RecognitionConfig(encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16,
										 sample_rate_hertz= 44100,
										 language_code = 'en-US',
										 enable_word_time_offsets = True,
										 model=model)

		#get response
	#	operation = client.long_running_recognize(config, audio)
		response = client.recognize(config, audio)
		
		confidence = np.mean([result.alternatives[0].confidence for result in response.results])
		
		# print urrent model's transcript
		# print('Model: ' + model)
		# print('Confidence: ' + str(confidence))
		# print('Transcribed Lyrics:\n')
		# for result in response.results:
		# 	print(result.alternatives[0].transcript, end=" ")
		# print('\n')
		
		if confidence > highest_confidence:
			highest_confidence = confidence
			chosen_response = response
			chosen_model = model
			
		#timeout if takes too long
	#	response = operation.result(timeout=90)

#	print("the type is: ", type(response))
	startEndTime = []
	transcript = []
	for result in response.results:
		wordList = result.alternatives[0].words
		transcript.append(result.alternatives[0].transcript)
		for wordInfo in wordList:
			startEndTime.append(convertToPython(wordInfo))
	transcript = [t[1:] if t[0] == " " else t for t in transcript]
	print('Transcribed Lyrics (' + chosen_model + '):\n' + ' '.join(transcript))

	return startEndTime, transcript

#separates lyrics into lines for display
def generateLines(audio, startEndTime):
	sr = 44100
	max_beat_pow = 8
	tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
	beat_times = librosa.frames_to_time(beats, sr=sr)
	start_times = [x["start"] for x in startEndTime]

	linesStartEnd = []
	lines = []

	# find how many words occur between different numbers of beats
	for loop_beats in [2**x for x in range(max_beat_pow+1)]:
		num_words = []
		curr_beat_times = beat_times[::loop_beats]
		# determine how many words start between x beats
		for i in range(len(curr_beat_times) - 1):
			num_words.append(len(list(filter(lambda x: x > curr_beat_times[i] and x < curr_beat_times[i+1], start_times))))
		if (num_words and max(num_words) >= 8) or loop_beats == 2**max_beat_pow:
			for i in range(len(curr_beat_times) - 1):

				# missing words at the beginning
				beginningStartEnd = list(filter(lambda x: x["start"] < curr_beat_times[i], startEndTime))
				beginning = ' '.join([i["word"] for i in list(filter(lambda x: x["start"] < curr_beat_times[i], startEndTime))])

				# words in current window of beats
				currStartEnd = list(filter(lambda x: x["start"] >= curr_beat_times[i] and x["start"] < curr_beat_times[i+1], startEndTime))
				curr = ' '.join([i["word"] for i in list(filter(lambda x: x["start"] >= curr_beat_times[i] and x["start"] < curr_beat_times[i+1], startEndTime))])

				# missing words at the end
				endStartEnd = list(filter(lambda x: x["start"] >= curr_beat_times[i+1], startEndTime))
				end = ' '.join([i["word"] for i in list(filter(lambda x: x["start"] >= curr_beat_times[i+1], startEndTime))])

				# check if there are any words at all each time
				# add missing words at the beginning
				if i == 0 and len(beginningStartEnd) > 0:
					linesStartEnd.append(beginningStartEnd)
					lines.append(beginning)

				# add words in current time range
				if len(currStartEnd) > 0:
					linesStartEnd.append(currStartEnd)
					lines.append(curr)

				# add missing words at the end
				if i == len(curr_beat_times) - 2 and len(endStartEnd) > 0:
					linesStartEnd.append(endStartEnd)
					lines.append(end)
			break

	return linesStartEnd, lines

#Approximate Algorithm returns a list of tuple of tuple ((start, end), lyric)
def approximate(lineStartEndTime):
	syllableStartEndTime = []
	#print(startEndTime)
	dic = pyphen.Pyphen(lang='en')
	for line in lineStartEndTime:
		lyricLine = [word["word"] for word in line]
		for wordPos, word in enumerate(line):	
			#split into syllables
			syllables = (dic.inserted(word["word"])).split("-")
			#find start and end times of the word
			start = word["start"]
			end = word["end"]
			timeDuration = end-start
			#calculate start and end of syllable times
			if(len(syllables) == 0):
				timeStep = timeDuration/float(1)
			else:
				timeStep = timeDuration/float(len(syllables))
			#find where the word occurs in the line
			#if its the last word, increment to the next line
			count = 0
			for i in range(len(syllables)):
				syll = syllables[i]
				syllStart = start + timeStep * count
				syllEnd = start + timeStep * (count+1)
				textEmphasis = " ".join(lyricLine[:wordPos] + ["".join(syllables[:i] + ["***" + syll + "***"] + syllables[i+1:])] + lyricLine[wordPos+1:])
				formattedData = ((syllStart, syllEnd), textEmphasis)
				syllableStartEndTime.append(formattedData)
	return syllableStartEndTime

#process wav file for onsets. Returns ((start, end), lyric)
def onsetModel(filepath, lineStartEndTime):
	sr = 44100
	syllModel = []
	dic = pyphen.Pyphen(lang='en')
	currLyricLine = 0
	for line in lineStartEndTime:
		sepLine = [wordStartEndTime['word'] for wordStartEndTime in line]
		for wordPos, word in enumerate(line):
			start = word['start']
			end = word['end']
			duration = end-start
			#get number of syllables, i.e. onsets, we want
			syllables = (dic.inserted(word["word"])).split("-")
			desiredOnsets = len(syllables)
			#if it is single syllable
			if desiredOnsets == 1 or duration == 0:
				emphasizedLine = " ".join(sepLine[:wordPos] + ["***" + syllables[0] +"***"] + sepLine[wordPos+1:])
				syllModel.append(((start, end), emphasizedLine))
		#if we have multiple syllables
			else:
				#get audio signal
				audio, sr = librosa.load(filepath, offset = start, duration = duration)
	
				#perform onset detection returns n onsets
				onsetFrames = librosa.onset.onset_detect(y = audio, sr = sr)
				
				#if the number of onsets is less than the desired n, we assume their length is the same
				if len(onsetFrames) < desiredOnsets:
					currStart = start
					stepSize = (end - start)/desiredOnsets
					for syll in syllables:
						syllModel.append(((currStart, currStart+stepSize), syll))
						currStart += stepSize

			#else, we need to find n highest peaks
				else:
				#compute onset strength on the section
					onsetStrengths = librosa.onset.onset_strength(y = audio, sr = sr)
				
					#corresponding strength to frames
					strengthToFrames = []
					for onsetFrame in onsetFrames:
						strengthToFrames.append((onsetStrengths[onsetFrame], onsetFrame))
					#sort in descending order
					sortedFrames = np.sort(strengthToFrames)[::1]
					#extract the highest frames and find the appropriate time
					nHighestFrames = [frame[1] for frame in sortedFrames[:desiredOnsets]]
					times = librosa.frames_to_time(nHighestFrames, sr = sr)
				
					for i in range(desiredOnsets-1):
						#calculate the start end times of the syllables
						syllStart = start + times[i]
						syllEnd = start+times[i+1]
						#calculate the emphasized lyrics
						emphasizedLine = " ".join(sepLine[:wordPos] + syllables[:i] +["***" + syllables[i] + "***"] + syllables[i+1:] + sepLine[wordPos+1:])
						syllModel.append(((syllStart, syllEnd), emphasizedLine))
					emphasizedLine = " ".join(sepLine[:wordPos] + syllables[:desiredOnsets-1] +["***" + syllables[desiredOnsets-1] + "***"] + sepLine[wordPos+1:])

					syllModel.append(((start + times[desiredOnsets-1], end), emphasizedLine))
	return syllModel
	
#finds the length of audio
def findLengthOfAudio(filepath):
	with contextlib.closing(wave.open(filepath, 'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)
	return duration

# plots an audio waveform with vertical lines at the syllables
def plot_audio_with_syllables(audio, syllables, title, word=None):
	dic = pyphen.Pyphen(lang='en')
	word_found = False
	sr = 44100
	lineheight = 1.5*max(np.abs(audio))
	num_syllables = len(syllables)
	length = float(audio.shape[0]) / sr

	t = np.linspace(0,length,audio.shape[0])

	plt.figure(figsize=(16,4))
	plt.plot(t, audio)
	plt.ylabel('Amplitude')
	plt.xlabel('Time (s)')
	plt.title(title)

	if word is not None:
		word_syll = (dic.inserted(word)).split("-")

	for i, syllable in enumerate(syllables):
		plt.axvline(syllable[0][0], -lineheight, lineheight, color='green', linestyle='--')
		# plt.axvline(syllable[0][1], -lineheight, lineheight, color='red')
		if word is not None and not word_found:
			try:
				if all("***"+word_syll[x]+"***" in syllables[x][1] for x in range(len(word_syll))):
					word_start = syllables[i][0][0]
					word_end = syllables[i+len(word_syll)][0][1]
					word_found = True
			except:
				pass

	if word_found:
		plt.xlim(word_start,word_end)
	
	plt.show()

#adds lyrics to the Sound Video File
def addSubtitles(startEndTimes, filepath, filename, vidLen):
	video = VideoFileClip(filepath)
	currVideo = video
	generator = lambda text: TextClip(text, font="Times", fontsize=150, color='white').set_pos('center')
	sub = SubtitlesClip(startEndTimes, generator)
	final = CompositeVideoClip([video, sub.set_pos(('center', 'center'))])
	path = './movieFiles/Karaoke/' + filename + '.avi'
	final.write_videofile(path, codec = 'libx264', audio_codec = 'pcm_s32le', fps = 24)

#Creates video with only audio
def makeVideoWithAudio(filepath, filename):
	#make blank black video
	audioLength = findLengthOfAudio(filepath)
	blankClip = ImageClip("./movieFiles/blackBackground.jpg", duration = audioLength)
	#initialize relevant audio variables
	song = mpy.AudioFileClip(filepath)
	overlayedClip = blankClip.set_audio(song)
	#output file
	path = './movieFiles/Karaoke/' + filename + '.avi'
	overlayedClip.write_videofile(path, codec = 'libx264', audio_codec = 'pcm_s32le', fps = 24)
	return path

#Helper function to wrun specific type of model
def runModel(songName, model):
	filepath = getFilePath(songName + ".wav")

	# graph audio waveform
	sr = 44100
	audio, sr = librosa.load(filepath,sr)
	# plot_audio(audio, sr, figsize=(16,4), title="I'm Yours")

	# execute transcription
#	startEndTime, lyrics = getSpeechInfo(filepath)
	startEndTime = [{'word': 'when', 'start': 0.0, 'end': 0.4}, {'word': 'you', 'start': 0.4, 'end': 0.5}, {'word': 'try', 'start': 0.5, 'end': 1.1}, {'word': 'your', 'start': 1.1, 'end': 1.5}, {'word': 'best', 'start': 1.5, 'end': 1.8}, {'word': 'but', 'start': 1.8, 'end': 2.3}, {'word': 'you', 'start': 2.3, 'end': 2.5}, {'word': "don't", 'start': 2.5, 'end': 2.7}, {'word': 'succeed', 'start': 2.7, 'end': 3.1}, {'word': 'when', 'start': 7.0, 'end': 7.5}, {'word': 'you', 'start': 7.5, 'end': 7.6}, {'word': 'get', 'start': 7.6, 'end': 8.0}, {'word': 'what', 'start': 8.0, 'end': 8.4}, {'word': 'you', 'start': 8.4, 'end': 8.7}, {'word': 'want', 'start': 8.7, 'end': 9.4}, {'word': 'but', 'start': 9.4, 'end': 9.6}, {'word': 'not', 'start': 9.6, 'end': 9.7}, {'word': 'what', 'start': 9.7, 'end': 10.1}, {'word': 'you', 'start': 10.1, 'end': 10.5}, {'word': 'need', 'start': 10.5, 'end': 10.7}, {'word': 'when', 'start': 13.6, 'end': 14.5}, {'word': 'you', 'start': 14.5, 'end': 14.6}, {'word': 'feel', 'start': 14.6, 'end': 15.2}, {'word': 'so', 'start': 15.2, 'end': 15.6}, {'word': 'tired', 'start': 15.6, 'end': 15.8}, {'word': 'but', 'start': 15.8, 'end': 16.5}, {'word': 'you', 'start': 16.5, 'end': 16.8}, {'word': "can't", 'start': 16.8, 'end': 17.0}, {'word': 'sleep', 'start': 17.0, 'end': 17.5}, {'word': 'stuck', 'start': 17.5, 'end': 20.0}, {'word': 'in', 'start': 20.0, 'end': 21.3}, {'word': 'reverse', 'start': 21.3, 'end': 21.6}, {'word': 'and', 'start': 27.7, 'end': 28.5}, {'word': 'the', 'start': 28.5, 'end': 28.6}, {'word': 'tears', 'start': 28.6, 'end': 29.2}, {'word': 'comes', 'start': 29.2, 'end': 29.8}]
	print(startEndTime)
#	startEndTime = [{'word': 'when', 'start': 0.0, 'end': 0.4}, {'word': 'you', 'start': 0.4, 'end': 0.5}, {'word': 'try', 'start': 0.5    , 'end': 1.0}, {'word': 'your', 'start': 1.0, 'end': 1.4}, {'word': 'best', 'start': 1.4, 'end': 1.7}, {'word': 'but', 'start': 1.7, 'end': 2.3}, {'word': 'you', 'start': 2.3, 'end': 2.5}, {'word': "don't", 'start': 2.5, 'end': 2.7}, {'word': 'succeed', 'start': 2.7, 'end': 3.1}, {'word': 'when', 'start': 7.0, 'end': 7.5}, {'word': 'you', 'start': 7.5, 'end': 7.6}, {'word': 'get', 'start': 7.6, 'end': 8.0}, {'word': 'what', 'start': 8.0, 'end': 8.4}, {'word': 'you', 'start': 8.4, 'end': 8.7}, {'word': 'want', 'start': 8.7, 'end': 9.4}, {'word': 'but'    , 'start': 9.4, 'end': 9.5}, {'word': 'not', 'start': 9.5, 'end': 9.8}, {'word': 'watching', 'start': 9.8, 'end': 10.5}, {'word': 'when', 'start': 14.0, 'end': 14.5}, {'word': 'you', 'start': 14.5, 'end': 14.6}, {'word': 'feel', 'start': 14.6, 'end': 14.9}, {'word': 'so', 'start': 14.9, 'end': 15.6}, {'word': 'tired', 'start': 15.6, 'end': 15.8}, {'word': 'but', 'start': 15.8, 'end': 16.6}, {'word': 'you', 'start': 16.6, 'end': 16.9}, {'word': "can't", 'start': 16.9, 'end': 17.0}, {'word': 'sleep', 'start': 17.0, 'end': 17.6}, {'word': 'cheer', 'start': 28.1, 'end': 29.2}, {'word': 'skirts', 'start': 29.2, 'end': 29.9}]
#	lyrics = ["when you try your best but you don't succeed", 'when you get what you want but not watching', "when you feel so tired but you can't sleep", 'cheer skirts']

	# separate into lines
	linesStartEnd, lines = generateLines(audio, startEndTime)
	print("Generated Lines:", lines)
#	print(linesStartEnd)

	if model == "onset":
		modelTimes = onsetModel(filepath, linesStartEnd)
	else:
		modelTimes = approximate(linesStartEnd)

	print(modelTimes)
#	plot_audio_with_syllables(audio, modelTimes, songName)
#	approx = [{'syll': 'when', 'start': 0.0, 'end': 0.4}, {'syll': 'you', 'start': 0.4, 'end': 0.5}, {'syll': 'try', 'start': 0.5, 'end': 1.0}, {'syll': 'your', 'start': 1.0, 'end': 1.4}, {'syll': 'best', 'start': 1.4, 'end': 1.7}, {'syll': 'but', 'start': 1.7, 'end': 2.3}, {'syll': 'you', 'start': 2.3, 'end': 2.5}, {'syll': "don't", 'start': 2.5, 'end': 2.7}, {'syll': 'suc', 'start': 2.7, 'end': 2.9000000000000004}, {'syll': 'ceed', 'start': 2.9000000000000004, 'end': 3.1}, {'syll': 'when', 'start': 7.0, 'end': 7.5}, {'syll': 'you', 'start': 7.5, 'end': 7.6}, {'syll': 'get', 'start': 7.6, 'end': 8.0}, {'syll': 'what', 'start': 8.0, 'end': 8.4}, {'syll': 'you', 'start': 8.4, 'end': 8.7}, {'syll': 'want', 'start': 8.7, 'end': 9.4}, {'syll': 'but', 'start': 9.4, 'end': 9.5}, {'syll': 'not', 'start': 9.5, 'end': 9.8}, {'syll': 'watch', 'start': 9.8, 'end': 10.15}, {'syll': 'ing', 'start': 10.15, 'end': 10.5}, {'syll': 'when', 'start': 14.0, 'end': 14.5}, {'syll': 'you', 'start': 14.5, 'end': 14.6}, {'syll': 'feel', 'start': 14.6, 'end': 14.9}, {'syll': 'so', 'start': 14.9, 'end': 15.6}, {'syll': 'tired', 'start': 15.6, 'end': 15.8}, {'syll': 'but', 'start': 15.8, 'end': 16.6}, {'syll': 'you', 'start': 16.6, 'end': 16.9}, {'syll': "can't", 'start': 16.9, 'end': 17.0}, {'syll': 'sleep', 'start': 17.0, 'end': 17.6}, {'syll': 'cheer', 'start': 28.1, 'end': 29.2}, {'syll': 'skirts', 'start': 29.2, 'end': 29.9}]
#	approx = [((0.0, 0.4), 'when'), ((0.4, 0.5), 'you'), ((0.5, 1.0), 'try'), ((1.0, 1.4), 'your'), ((1.4, 1.7), 'best'), ((1.7, 2.3), 'but'), ((2.3, 2.5), 'you'), ((2.5, 2.7), "don't"), ((2.7, 2.9000000000000004), 'suc'), ((2.9000000000000004, 3.1), 'ceed'), ((7.0, 7.5), 'when'), ((7.5, 7.6), 'you'), ((7.6, 8.0), 'get'), ((8.0, 8.4), 'what'), ((8.4, 8.7), 'you'), ((8.7, 9.4), 'want'), ((9.4, 9.5), 'but'), ((9.5, 9.8), 'not'), ((9.8, 10.15), 'watch'), ((10.15, 10.5), 'ing'), ((14.0, 14.5), 'when'), ((14.5, 14.6), 'you'), ((14.6, 14.9), 'feel'), ((14.9, 15.6), 'so'), ((15.6, 15.8), 'tired'), ((15.8, 16.6), 'but'), ((16.6, 16.9), 'you'), ((16.9, 17.0), "can't"), ((17.0, 17.6), 'sleep'), ((28.1, 29.2), 'cheer'), ((29.2, 29.9), 'skirts')] 
#	print(modelTimes)
#	print("start end time", startEndTime)
	# videoFile = makeVideoWithAudio(filepath, songName)	
	# videoFile = './movieFiles/Karaoke/' + songName+ '.avi'
	# videoLen = findLengthOfAudio(filepath)
	# addSubtitles(modelTimes, videoFile, songName +"AnnotatedVid" + model, videoLen)


def main():
	#load Google Credentials
	loadCredentials()
	#Get console arguments
	args = sys.argv
	songName = args[1]
	model = args[2]
	#run the desired Model
	runModel(songName, model)
main()

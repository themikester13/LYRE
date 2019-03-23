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
	# print('Transcribed Lyrics (' + chosen_model + '):\n' + ' '.join(transcript))

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
				count += 1
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
			#if duration is 0
			if duration == 0 or desiredOnsets == 1:
				emphasizedLine = " ".join(sepLine[:wordPos] + ["***" + word["word"] +"***"] + sepLine[wordPos+1:])
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
					for i, syll in enumerate(syllables):
						emphasizedLine = " ".join(sepLine[:wordPos] + syllables[:i] + ["***" + syll +"***"] +syllables[i+1:] + sepLine[wordPos+1:])
						syllModel.append(((start, end), emphasizedLine))
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
def plot_audio_with_syllables(audio, syllables, words, title, model, input_word=None):
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

	for syllable in syllables:
		plt.axvline(syllable[0][0], -lineheight, lineheight, color='green', linestyle='--')
		# plt.axvline(syllable[0][1], -lineheight, lineheight, color='red')

	for word in words:
		if word['word'] == input_word:
			start = word['start']
			end = word['end']
			duration = end-start
			plt.xlim(start-0.25*duration, end+0.25*duration)
			plt.axvline(start, -lineheight, lineheight, color='black', linewidth=4)
			plt.axvline(end, -lineheight, lineheight, color='black', linewidth=4)
			title = title + " - '" + input_word + "'"
			break
	
	plt.title(title + " (" + model + ")")
	plt.show()

#processes output of start end time for better display
def newEndTime(startEndTime):
	#print(startEndTime[:-1])
	replacedEndTimes = []
	for i, syll in enumerate(startEndTime[:-1]):
		replacedEndTimes.append(((syll[0][0], startEndTime[i+1][0][0]), syll[1]))
	replacedEndTimes.append(startEndTime[-1])	
	return replacedEndTimes

#adds lyrics to the Sound Video File
def addSubtitles(startEndTimes, filepath, filename, vidLen):
	video = VideoFileClip(filepath)
	currVideo = video
	generator = lambda text: TextClip(text, font="Times", fontsize=150, color='white').set_pos('center')
	sub = SubtitlesClip(newEndTime(startEndTimes), generator)
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

def splitAudio(fullFile, newTime, max_length):
	file_num = newTime[0]//(max_length*10**3)
	newAudio = AudioSegment.from_wav(fullFile)
	newAudio = newAudio[newTime[0]:newTime[1]]
	newFile = fullFile[:-4] + '_' + str(file_num) + '.wav'
	newAudio.export(newFile, format="wav")
	return newFile

# Helper function to run specific type of models and deal with audio length issues
def runModel(songName, model, graph_word=None):
	sr = 44100

	# initialize data structures
	startEndTime = []
	lyrics = []

	# maximum length of an audio file we cna handle at a time (seconds)
	max_length = 59

	# get filepath from song name
	filepath = getFilePath(songName + ".wav")

	# load audio
	audio, sr = librosa.load(filepath,sr)

	# if length of audio is over the max length, split it up
	length = findLengthOfAudio(filepath)
	iterations = math.ceil(length/max_length)

	# print song information
	print('Song name: "' + songName + '"')
	print('Song length:', length, 'seconds')
	print('Syllable model:', model)

	# transcribe lyrics for each chunk
	print('Waiting for transcription to complete...')
	for i in range(iterations):
		# compute start and end times of audio
		start_time = max_length*i
		if i == iterations - 1:
			end_time = length
		else:
			end_time = max_length*(i+1)

		# split audio
		curr_filepath = splitAudio(filepath, (10**3*start_time,10**3*end_time), max_length)
		curr_audio, sr = librosa.load(curr_filepath,sr)

		# execute transcription
		curr_startEndTime, curr_lyrics = getSpeechInfo(curr_filepath)

		# adjust start times to account for splitting up of audio
		for curr in curr_startEndTime:
			curr['start'] += start_time
			curr['end'] += start_time

		# add to overall lists
		startEndTime += curr_startEndTime
		lyrics += curr_lyrics

		# delete split file after use
		os.remove(curr_filepath)
	lyrics = ' '.join(lyrics)
	print("Transcribed Lyrics:", lyrics)

	# separate into lines
	print('Generating lines...')
	linesStartEnd, lines = generateLines(audio, startEndTime)
	# print("Generated Lines:", lines)

	# separate syllables
	print('Separating syllables...')
	if model == "onset":
		modelTimes = onsetModel(filepath, linesStartEnd)
	else:
		modelTimes = approximate(linesStartEnd)
	# print("Separated Syllables:", modelTimes)

	# graph waveform
	plot_audio_with_syllables(audio, modelTimes, startEndTime, songName, model, graph_word)

	# generate video
	# print('Generating video...')
	# videoFileName = './movieFiles/Karaoke/' + songName + '.avi'
	# if os.path.isfile(videoFileName):
	# 	videoFile = videoFileName
	# else:
	# 	videoFile = makeVideoWithAudio(filepath, songName)
	# addSubtitles(modelTimes, videoFile, songName +"AnnotatedVid" + model, length)
	# print('Video saved as: ' + videoFile + "AnnotatedVid" + model)

def main():
	#load Google credentials
	loadCredentials()
	#Get console arguments
	args = sys.argv
	songName = args[1]
	model = args[2]
	if len(args) == 4:
		graph_word = args[3]
	else:
		graph_word = None
	#run the desired Model
	runModel(songName, model, graph_word)
main()

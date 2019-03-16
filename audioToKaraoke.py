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
import numpy as np, scipy as sp, sklearn, librosa, cmath,math
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
#	print(wordInfo.word, wordInfo.start_time.seconds)
	startEnd["word"] = wordInfo.word
	startEnd["start"] = durationToSec(wordInfo.start_time)
	startEnd["end"] = durationToSec(wordInfo.end_time)
	return startEnd

#makes API Call
def getSpeechInfo(path2File):
	#gets path to file. Assumes its in soundfiles/foreground/ directory	
	client = speech.SpeechClient()
	
	#loading audio into proper file
#	stereoToMono(path2File)
	with io.open(path2File, 'rb') as audio_file:
		content = audio_file.read()
		audio = types.RecognitionAudio(content = content)
	
	config = types.RecognitionConfig(encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz= 44100, language_code = 'en-US', enable_word_time_offsets = True)

	#get response
#	operation = client.long_running_recognize(config, audio)
	response = client.recognize(config, audio)	

	#timeout if takes too long
#	print('Waiting for operation to complete...')
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
	return startEndTime, transcript

#Approximate Algorithm returns a list of tuple of tuple ((start, end), lyric)
def approximate(startEndTime, lyric):
	syllableStartEndTime = []
	#print(startEndTime)
	dic = pyphen.Pyphen(lang='en')
	currLyricLine = 0
	wordPosition = 0
	nextLine = False
	for word in startEndTime:	
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
		#the list of words in the line
		lyricLine = lyric[currLyricLine].split(" ")
		#find where the word occurs in the line
		#if its the last word, increment to the next line
		if wordPosition == len(lyricLine) - 1:
			currLyricLine += 1
			nextLine = True
		count = 0
		for i in range(len(syllables)):
			syll = syllables[i]
			syllStart = start + timeStep * count
			syllEnd = start + timeStep * (count+1)
			textEmphasis = " ".join(lyricLine[:wordPosition] + ["".join(syllables[:i] + ["***" + syll + "***"] + syllables[i+1:])] + lyricLine[wordPosition+1:])
			formattedData = ((syllStart, syllEnd), textEmphasis)
			syllableStartEndTime.append(formattedData)
			count += 1
		if nextLine:
			nextLine = False
			wordPosition = 0
		else:
			wordPosition += 1
	return syllableStartEndTime

#process wav file for onsets. Returns ((start, end), lyric)
def onsetModel(filepath, wordStartEndTime, lyrics):
	sr = 44100
	syllModel = []
	dic = pyphen.Pyphen(lang='en')
	currLyricLine = 0
	moveToNextLine = False
	wordPos = 0
	for word in wordStartEndTime:
		start = word['start']
		end = word['end']	
		#get number of syllables, i.e. onsets, we want
		syllables = (dic.inserted(word["word"])).split("-")
		desiredOnsets = len(syllables)
		currLine = lyrics[currLyricLine].split(" ")
		if wordPos == len(currLine) - 1:
			moveToNextLine = True
		#if it is single syllable
		if desiredOnsets == 1:
			emphasizedLine = " ".join(currLine[:wordPos] + ["***" + syllables[0] +"***"] + currLine[wordPos+1:])
			syllModel.append(((start, end), emphasizedLine))
		#if we have multiple syllables
		else:
			#get audio signal
			audio, sr = librosa.load(filepath, offset = start, duration = end-start)
	
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
					emphasizedLine = " ".join(currLine[:wordPos] + syllables[:i] +["***" + syllables[i] + "***"] + syllables[i+1:] + currLine[wordPos+1:])
					syllModel.append(((syllStart, syllEnd), emphasizedLine))
				emphasizedLine = " ".join(currLine[:wordPos] + syllables[:desiredOnsets-1] +["***" + syllables[desiredOnsets-1] + "***"] + currLine[wordPos+1:])

				syllModel.append(((start + times[desiredOnsets-1], end), emphasizedLine))
		#if we are moving to next line, move to next line and reset the counter
		if moveToNextLine:
			moveToNextLine = False
			currLyricLine += 1
			wordPos = 0
		else:
			wordPos += 1			
	return syllModel
	
#finds the length of audio
def findLengthOfAudio(filepath):
	with contextlib.closing(wave.open(filepath, 'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)
	return duration

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
	startEndTime, lyrics = getSpeechInfo(filepath)
#	print(startEndTime)
#	startEndTime = [{'word': 'when', 'start': 0.0, 'end': 0.4}, {'word': 'you', 'start': 0.4, 'end': 0.5}, {'word': 'try', 'start': 0.5    , 'end': 1.0}, {'word': 'your', 'start': 1.0, 'end': 1.4}, {'word': 'best', 'start': 1.4, 'end': 1.7}, {'word': 'but', 'start': 1.7, 'end': 2.3}, {'word': 'you', 'start': 2.3, 'end': 2.5}, {'word': "don't", 'start': 2.5, 'end': 2.7}, {'word': 'succeed', 'start': 2.7, 'end': 3.1}, {'word': 'when', 'start': 7.0, 'end': 7.5}, {'word': 'you', 'start': 7.5, 'end': 7.6}, {'word': 'get', 'start': 7.6, 'end': 8.0}, {'word': 'what', 'start': 8.0, 'end': 8.4}, {'word': 'you', 'start': 8.4, 'end': 8.7}, {'word': 'want', 'start': 8.7, 'end': 9.4}, {'word': 'but'    , 'start': 9.4, 'end': 9.5}, {'word': 'not', 'start': 9.5, 'end': 9.8}, {'word': 'watching', 'start': 9.8, 'end': 10.5}, {'word': 'when', 'start': 14.0, 'end': 14.5}, {'word': 'you', 'start': 14.5, 'end': 14.6}, {'word': 'feel', 'start': 14.6, 'end': 14.9}, {'word': 'so', 'start': 14.9, 'end': 15.6}, {'word': 'tired', 'start': 15.6, 'end': 15.8}, {'word': 'but', 'start': 15.8, 'end': 16.6}, {'word': 'you', 'start': 16.6, 'end': 16.9}, {'word': "can't", 'start': 16.9, 'end': 17.0}, {'word': 'sleep', 'start': 17.0, 'end': 17.6}, {'word': 'cheer', 'start': 28.1, 'end': 29.2}, {'word': 'skirts', 'start': 29.2, 'end': 29.9}]
#	lyrics = ["when you try your best but you don't succeed", 'when you get what you want but not watching', "when you feel so tired but you can't sleep", 'cheer skirts']

	if model == "onset":
		modelTimes = onsetModel(filepath, startEndTime, lyrics)
	else:
		modelTimes = approximate(startEndTime, lyrics)
#	print(modelTimes)
#	approx = [{'syll': 'when', 'start': 0.0, 'end': 0.4}, {'syll': 'you', 'start': 0.4, 'end': 0.5}, {'syll': 'try', 'start': 0.5, 'end': 1.0}, {'syll': 'your', 'start': 1.0, 'end': 1.4}, {'syll': 'best', 'start': 1.4, 'end': 1.7}, {'syll': 'but', 'start': 1.7, 'end': 2.3}, {'syll': 'you', 'start': 2.3, 'end': 2.5}, {'syll': "don't", 'start': 2.5, 'end': 2.7}, {'syll': 'suc', 'start': 2.7, 'end': 2.9000000000000004}, {'syll': 'ceed', 'start': 2.9000000000000004, 'end': 3.1}, {'syll': 'when', 'start': 7.0, 'end': 7.5}, {'syll': 'you', 'start': 7.5, 'end': 7.6}, {'syll': 'get', 'start': 7.6, 'end': 8.0}, {'syll': 'what', 'start': 8.0, 'end': 8.4}, {'syll': 'you', 'start': 8.4, 'end': 8.7}, {'syll': 'want', 'start': 8.7, 'end': 9.4}, {'syll': 'but', 'start': 9.4, 'end': 9.5}, {'syll': 'not', 'start': 9.5, 'end': 9.8}, {'syll': 'watch', 'start': 9.8, 'end': 10.15}, {'syll': 'ing', 'start': 10.15, 'end': 10.5}, {'syll': 'when', 'start': 14.0, 'end': 14.5}, {'syll': 'you', 'start': 14.5, 'end': 14.6}, {'syll': 'feel', 'start': 14.6, 'end': 14.9}, {'syll': 'so', 'start': 14.9, 'end': 15.6}, {'syll': 'tired', 'start': 15.6, 'end': 15.8}, {'syll': 'but', 'start': 15.8, 'end': 16.6}, {'syll': 'you', 'start': 16.6, 'end': 16.9}, {'syll': "can't", 'start': 16.9, 'end': 17.0}, {'syll': 'sleep', 'start': 17.0, 'end': 17.6}, {'syll': 'cheer', 'start': 28.1, 'end': 29.2}, {'syll': 'skirts', 'start': 29.2, 'end': 29.9}]
#	approx = [((0.0, 0.4), 'when'), ((0.4, 0.5), 'you'), ((0.5, 1.0), 'try'), ((1.0, 1.4), 'your'), ((1.4, 1.7), 'best'), ((1.7, 2.3), 'but'), ((2.3, 2.5), 'you'), ((2.5, 2.7), "don't"), ((2.7, 2.9000000000000004), 'suc'), ((2.9000000000000004, 3.1), 'ceed'), ((7.0, 7.5), 'when'), ((7.5, 7.6), 'you'), ((7.6, 8.0), 'get'), ((8.0, 8.4), 'what'), ((8.4, 8.7), 'you'), ((8.7, 9.4), 'want'), ((9.4, 9.5), 'but'), ((9.5, 9.8), 'not'), ((9.8, 10.15), 'watch'), ((10.15, 10.5), 'ing'), ((14.0, 14.5), 'when'), ((14.5, 14.6), 'you'), ((14.6, 14.9), 'feel'), ((14.9, 15.6), 'so'), ((15.6, 15.8), 'tired'), ((15.8, 16.6), 'but'), ((16.6, 16.9), 'you'), ((16.9, 17.0), "can't"), ((17.0, 17.6), 'sleep'), ((28.1, 29.2), 'cheer'), ((29.2, 29.9), 'skirts')] 
#	print(modelTimes)
#	print("start end time", startEndTime)
	videoFile = makeVideoWithAudio(filepath, songName)	
#	videoFile = './movieFiles/Karaoke/' + songName+ '.avi'
	videoLen = findLengthOfAudio(filepath)
	addSubtitles(modelTimes, videoFile, songName +"AnnotatedVid" + model, videoLen)


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
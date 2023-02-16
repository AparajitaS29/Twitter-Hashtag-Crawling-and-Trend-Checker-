from flask import Flask, render_template, request
from youtube_search import YoutubeSearch
import requests
import json
import pathlib
from pymongo import MongoClient
import yaml
from flask_cors import CORS
import jsonify
from textblob import TextBlob

app = Flask(__name__)
#database connection
client=MongoClient()

# Connect with the portnumber and host
client = MongoClient("mongodb://localhost:27017/")
  
# Access database
mydatabase = client["example_db"]

# Access collection of the database
myCollection = mydatabase["IR2"]



@app.route("/", methods=['GET'])
def home():
	return render_template("index.html")

@app.route("/out")
def out():
	return render_template("template.html")


@app.route("/out", methods=['POST', 'GET'])
def search():
	if request.method == 'POST':
		allData = myCollection.find()
		dataJson = []
		for data in allData:
			id = data['_id']
			keyword = data["word"]
			if keyword == "covid":
				tweets = data['tweet']
			#print(keyword,tweets)
			dataDict = {
                "tweets": tweets
            }
			dataJson.append(dataDict)
	#print(dataJson)
	for i in range(len(dataJson[0]["tweets"])):
		res = TextBlob(dataJson[0]["tweets"][i])
		score=res.sentiment.polarity
		val=0
		if score < 0:
			val="Negative"
			dataJson[0]["tweets"][i]=dataJson[0]["tweets"][i]+" - "+str(val)
		elif score == 0:
			val ="Neutral"
			dataJson[0]["tweets"][i]=dataJson[0]["tweets"][i]+" - "+str(val)
		else:
			val ="Positive"
			dataJson[0]["tweets"][i]=dataJson[0]["tweets"][i]+" - "+str(val)

	#return str(dataJson)

	request_json = request.get_json()
	term= request_json['search']
	#term="key error python"
	to_be_written=""
	#print(pathlib.Path().absolute())
	f= open("/home/donnie/pythonui/google_search/templates/template.html","w")
	start= '''<table id="output" style="height: 317px; width: 100%;">
				<tbody>
				<tr style="height: 109px;">
				</tr>'''
	end= '''</tbody>
			</table>'''
	f.write(start)
	for i in range(0,5):
		#r1 = "covid"
		sr = dataJson[0]["tweets"][i]
		print(sr)
		temp=''
		temp+='<tr style="height: 108px;">'
		temp+='<td style="width: 50%; height: 108px;">'
		#temp+='<p style="text-align: center;"><a href="{}" target="_blank" rel="noopener">{}</a></p>'.format(sr['link'], sr['title'].encode('utf8'))
		#temp+='<p style="text-align: center;"><strong>Search Word</strong>-&nbsp;{}<br /><strong>Tweet</strong>-&nbsp;{}</p>'.format(sr,sr)
		temp+='<p style="text-align: center;"><strong>Search Word</strong>-&nbsp;{}</p>'.format(sr)
		temp+='</td>'
		#print(temp)
		f.write(temp)
	f.write(end)
	f.close()
	return "done"


if __name__ == "__main__":
	app.run()
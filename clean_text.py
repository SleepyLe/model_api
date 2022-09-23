import re
import demoji

stop_word=[]
txt_file = open("vietnamese-stopwords-dash.txt", "r", encoding="utf8")
file_content = txt_file.read()
content_list = file_content.split("\n")

def remove_url(text):
    text = re.sub(r"http\S+", "", text)
    return text


def remove_stopwords(text):
    text = [word for word in text if word not in content_list]
    return text

def handle_emoji(string):
    emojis = demoji.findall(string)

    for emoji in emojis:
        string = string.replace(emoji, " " + emojis[emoji].split(":")[0])

    return string
from flask import Flask, request, jsonify
import urllib.request
import json


app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_file_content():
    # Получаем ссылку на файл от клиента
    url = request.form['file_url']

    # Открываем файл по ссылке и получаем его содержимое
    #with urllib.request.urlopen(file_url) as f:
        #file_content = f.read().decode('utf-8')

    # Возвращаем содержимое файла в ответ на запрос клиента
    #return file_content
    url = 'https://karton.na4u.ru/1.txt'

     #with urllib.request.urlopen(url) as f:
         #text = f.read()  # получаем текст из файла, на который ссылается ссылка
    #file_content = ""

    try:
       response = urllib.request.urlopen(url)
       file_content = response.read().decode('utf-8')
       oo =file_content
    except urllib.error.HTTPError as e:
       oo = "HTTP Error: {e.code}, {e.reason}"
    except urllib.error.URLError as e:
       oo ="URL Error: {e.reason}"
    except Exception as e:
       oo ="Error: {e}"

    return oo


if __name__ == '__main__':
    app.run(debug=True)

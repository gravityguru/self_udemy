from flask import Flask

app = Flask(__name__)#__main__

@app.route('/home')

def greeting():
    return 'Welcome! Mr. Gurunayk'

if __name__ == '__main__':
    app.run(port = 3000)

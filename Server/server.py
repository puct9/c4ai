"""
2018 April fools
"""
from flask import Flask, Response, render_template

from engine_wrapper import EngineInstance


app = Flask(__name__)
app.secret_key = 'I DO NOT REALLY CARE ABOUT THIS REALLY NOT GONNA LIE'
app.send_file_max_age_default = 0

engine = EngineInstance('Engine', 'C4UCT.exe')


@app.route('/', methods=['GET'])
def index():
    return render_template('game.html')


@app.route('/eng/<p1>/<p2>/<p3>/<p4>/<p5>/<p6>', methods=['GET'])
def eng_compute(p1, p2, p3, p4, p5, p6):
    engine.setpos(f'{p1}/{p2}/{p3}/{p4}/{p5}/{p6}')
    return Response(engine.geteval(), mimetype='text/plain')


@app.route('/eng/<p1>/<p2>/<p3>/<p4>/<p5>/<p6>/<int:n>', methods=['GET'])
def eng_compute_n(p1, p2, p3, p4, p5, p6, n):
    n = min(30000, max(10, n))
    engine.setpos(f'{p1}/{p2}/{p3}/{p4}/{p5}/{p6}')
    return Response(engine.geteval(n), mimetype='text/plain')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8421, debug=True)

from flask import Flask, session, request, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rc
from sklearn.linear_model import LinearRegression
from datetime import datetime
import re, time, os, cStringIO
import healthgraph


app = Flask('RunKeeper Shoes')
app.config.from_pyfile('./config.py')


@app.route('/')
def index():
    if session.has_key('rk_access_token'):
        return redirect(url_for('predict'))
    else:
        rk_auth_mgr = healthgraph.AuthManager(app.config['CLIENT_ID'],
                                              app.config['CLIENT_SECRET'],
                                              '/'.join((app.config['BASEURL'], 'login',)))
        rk_auth_uri = rk_auth_mgr.get_login_url()
        rk_button_img = rk_auth_mgr.get_login_button_url('blue', 'black', 300)
    return """
<html>
    <body>
        <h1>Welcome to the Runkeeper Shoe Replacement Predictor!</h1>
        <p>
            <a href="%s">
	        <img src="%s" alt="RunKeeper Login Button">
	    </a>
	</p>
    </body>
</html>
           """ % (rk_auth_uri, rk_button_img)


@app.route('/login', methods=['GET'])
def login():
    code = request.args['code']
    if code is not None:
        rk_auth_mgr = healthgraph.AuthManager(app.config['CLIENT_ID'],
                                              app.config['CLIENT_SECRET'],
                                              '/'.join((app.config['BASEURL'], 'login',)))
        access_token = rk_auth_mgr.get_access_token(code)
        session['rk_access_token'] = access_token
        return redirect(url_for('predict'))


@app.route('/predict', methods=['GET'])
def predict():

    if not session.has_key('rk_access_token'):
        return redirect(url_for('index'))

    if 'bought' not in request.args or \
       not re.match('^\d{4}-\d{2}-\d{2}$', request.args['bought']):
        return """
<html>
    <body>
        <form action="predict" method="get">
        You bought your running shoes on: <input type="date" name="bought"><br>
        Only use date after (optional, if not set will use everything): <input type="date" name="start"><br>
        <input type="submit">
        </form>
    </body>
</html>
               """

    def to_dt(s):
        if ':' in s:
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        else:
            return datetime.strptime(s, '%Y-%m-%d')

    def to_ts(s):
        return time.mktime(to_dt(s).timetuple()) - ts0

    def from_ts(ts):
        return datetime.fromtimestamp(ts + ts0).strftime('%Y-%m-%d')

    data_start_date = None
    if 'start' in request.args and re.match('^\d{4}-\d{2}-\d{2}$', request.args['start']):
        data_start_date = request.args['start']
    buy_date = request.args['bought']

    low = 500
    high = 800

    user = healthgraph.User(session=healthgraph.Session(session['rk_access_token']))
    act_iter = user.get_fitness_activity_iter()

    dates, dists = zip(*[(act['start_time'], act['total_distance']) for act in act_iter if act['type'] == 'Running'])
    df = pd.DataFrame({'dates': dates, 'dists': dists})

    if data_start_date:
        df = df[df.dates >= data_start_date]

    df.sort('dates', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['cum_dists'] = df.dists.cumsum()
    buy_cum_dist = df[df.dates >= buy_date].min().cum_dists
    df['buy_dists'] = df.cum_dists - buy_cum_dist

    dates = [to_dt(str(d)) for d in df.dates]
    timestamps = np.asarray([time.mktime(d.timetuple()) for d in dates])
    ts0 = timestamps[0]
    timestamps -= ts0

    lr = LinearRegression(fit_intercept=False).fit(timestamps.reshape((df.shape[0], 1)), df.cum_dists)

    latest_x = ((high + 100) + buy_cum_dist) / lr.coef_
    X = np.linspace(timestamps[0], latest_x)
    Y = X * lr.coef_ - buy_cum_dist

    low_x = (low + buy_cum_dist) / lr.coef_
    high_x = (high + buy_cum_dist) / lr.coef_

    fig = plt.figure(figsize=(12, 7))

    points = plt.scatter(timestamps, df.buy_dists, color='b', label='Running sessions (cumulative distance)')
    line, = plt.plot(X, Y, linestyle='--', color='r', label='Linear regression')

    plt.axhspan(low, high, facecolor='y', alpha=0.25)
    plt.axvspan(low_x, high_x, facecolor='y', alpha=0.25)

    n_ticks = 20
    if to_dt(buy_date) >= df.dates.min():
        x_ts = np.linspace(X[0], X[-1], n_ticks)
    else:
        x_ts = np.linspace(to_ts(buy_date) - 1000000, X[-1], n_ticks)
    plt.xticks(x_ts, [from_ts(ts) for ts in x_ts], rotation=45, fontsize=10)
    plt.xlim(x_ts[0] - 5000000, x_ts[-1] + 5000000)

    buy = plt.axhline(0, linestyle=':', color='k', alpha=0.5)
    plt.axvline(to_ts(buy_date), linestyle=':', color='k', alpha=0.5)

    plt.ylabel('Total distance ran (in kms) since shoes were bought')

    # activate latex text rendering
    os.environ['PATH'] += ':/usr/texbin'
    rc('text', usetex=False)

    zone = Rectangle((0, 0), 1, 1, fc='y', alpha=0.25, fill=True, linewidth=1)
    plt.legend([buy, points, line, zone],
               ['Shoes bought (on %s)' % buy_date,
                'Running sessions (cumulative distance)',
                'Linear regression',
                'Replacement zone (from %s to %s)' % (from_ts(low_x), from_ts(high_x))],
               loc='upper right')

    sio = cStringIO.StringIO()
    fig.savefig(sio, format='png')

    return """
<html>
   <body>
       <h2>Shoe Replacement Prediction for %s</h2>
       <p><b>Here's how it works</b>: your running data (date and distance for every running session you have tracked)
          is first downloaded from RunKeeper
          (with the <a href="http://developer.runkeeper.com/healthgraph" target="_blank">Health Graph API</a>).
          The distances are then transformed into cumulative distances, and adjusted in such a way that the day you bought
          your shoes is roughly aligned with a distance of 0. These values are then "fitted" (imperfectly) using a
          <a href="http://cjauvin.blogspot.ca/2013/10/linear-regression-101.html" target="_blank">linear regression</a>,
          from which the date range where your shoes are predicted to have
          accumulated 500 to 800 kilometers of usage can be read (note that these values have been unscientifically gathered and averaged
          from sources like <a href="http://www.runnersworld.com/running-shoes/running-shoe-faq?page=single" target="_blank">this</a>).
       </p>
       <img src="data:image/png;base64,%s"/>
   </body>
</html>
           """ % (user.get_profile().get('name'),
                  sio.getvalue().encode("base64").strip())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=81)

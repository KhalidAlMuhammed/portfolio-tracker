from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
from dateutil import parser
import csv
from io import StringIO
import threading
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import seaborn as sns
import matplotlib.pyplot as plt
import schedule
import pytz
from datetime import datetime, time as datetime_time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

def create_database():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stocks
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  shares REAL,
                  price_per_share REAL,
                  total_price REAL,
                  transaction_type TEXT,
                  transaction_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS hourly_values
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  total_value REAL)''')
    conn.commit()
    conn.close()

def get_portfolio():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("""
        SELECT 
            symbol, 
            SUM(CASE WHEN transaction_type='buy' THEN shares ELSE -shares END) as total_shares,
            SUM(CASE WHEN transaction_type='buy' THEN total_price ELSE -total_price END) as total_cost
        FROM stocks 
        GROUP BY symbol 
        HAVING total_shares > 0
    """)
    portfolio = c.fetchall()
    conn.close()
    return portfolio


def calculate_portfolio_value(time_period="1d"):
    portfolio = get_portfolio()
    total_value = 0
    total_cost = 0
    period_start_value = 0
    portfolio_data = []
    for stock in portfolio:
        symbol, shares, cost = stock
        current_price, start_price = get_stock_price(symbol, time_period)
        if current_price is not None and start_price is not None:
            market_value = shares * current_price
            period_start_value += shares * start_price
            total_value += market_value
            total_cost += cost
            portfolio_data.append({
                'symbol': symbol,
                'shares': shares,
                'current_price': current_price,
                'market_value': market_value,
                'cost': cost,
                'gain_loss': market_value - cost,
            })
    total_gain_loss = total_value - total_cost
    period_gain_loss = total_value - period_start_value
    return portfolio_data, total_value, total_cost, total_gain_loss, period_gain_loss

def add_transaction(symbol, shares, price_per_share, total_price, transaction_type):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("INSERT INTO stocks (symbol, shares, price_per_share, total_price, transaction_type, transaction_date) VALUES (?, ?, ?, ?, ?, ?)",
              (symbol, shares, price_per_share, total_price, transaction_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
def get_portfolio_value():
    portfolio = get_portfolio()
    total_value = 0
    for stock in portfolio:
        symbol, shares, _ = stock
        current_price, _ = get_stock_price(symbol, "1d")
        if current_price is not None:
            total_value += shares * current_price
    return total_value
def get_portfolio_value_at_time(time_str):
    portfolio = get_portfolio()
    total_value = 0
    for stock in portfolio:
        symbol, shares, _ = stock
        current_price, _ = get_stock_price(symbol, "1d", time_str)
        if current_price is not None:
            total_value += shares * current_price
    return total_value

def get_stock_price(symbol, period="1d", time_str=None):
    try:
        ticker = yf.Ticker(symbol)
        if time_str:
            # Parse the time string to handle different formats
            parsed_time = parser.parse(time_str, fuzzy=True)
            # Convert to the correct format for yfinance
            start_time_str = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = (parsed_time + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")  # Add 1 minute to get a tiny range
            hist = ticker.history(start=start_time_str, end=end_time_str)
            if not hist.empty:
                return hist['Close'].iloc[-1], hist['Close'].iloc[0]
            else:
                return None, None
        else:
            hist = ticker.history(period=period)
            return hist['Close'].iloc[-1], hist['Close'].iloc[0]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

def get_historical_values(period):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    if period == "1d":
        c.execute("SELECT * FROM hourly_values WHERE timestamp >= datetime('now', '-1 day') ORDER BY timestamp")
    elif period == "1mo":
        c.execute("SELECT * FROM hourly_values WHERE timestamp >= datetime('now', '-1 month') ORDER BY timestamp")
    elif period == "1y":
        c.execute("SELECT * FROM hourly_values WHERE timestamp >= datetime('now', '-1 year') ORDER BY timestamp")
    elif period == "5y":
        c.execute("SELECT * FROM hourly_values WHERE timestamp >= datetime('now', '-5 years') ORDER BY timestamp")
    else:
        c.execute("SELECT * FROM hourly_values ORDER BY timestamp")
    data = c.fetchall()
    conn.close()
    return data

@app.route('/save_current_value', methods=['POST'])
def save_current_value():
    total_value = get_portfolio_value()
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("INSERT INTO hourly_values (timestamp, total_value) VALUES (?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), total_value))
    conn.commit()
    conn.close()
    flash('Current portfolio value saved successfully', 'success')
    return redirect(url_for('index'))

@app.route('/')
def index():
    portfolio_data, total_value, total_cost, total_gain_loss, _ = calculate_portfolio_value("1d")
    
    historical_data = get_historical_values("all")
    df = pd.DataFrame(historical_data, columns=['id', 'timestamp', 'value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    fig = px.line(df, x='timestamp', y='value', title='Portfolio Value Over Time')
    graph_json = fig.to_json()
    
    return render_template('index.html', portfolio=portfolio_data, total_value=total_value, 
                           total_cost=total_cost, total_gain_loss=total_gain_loss, 
                           graph_json=graph_json, start_date=start_date, end_date=end_date)

def insert_dummy_data():
    dummy_transactions = [
        ("AAPL", 10, 150.00, 1500.00, "buy", "2023-08-26 14:00:00"),
        ("GOOGL", 5, 1400.00, 7000.00, "buy", "2023-07-26 10:00:00"),
        ("MSFT", 15, 200.00, 3000.00, "buy", "2022-08-26 09:00:00"),
        ("TSLA", 8, 750.00, 6000.00, "buy", "2024-01-26 14:00:00"),
        ("AMZN", 2, 3200.00, 6400.00, "buy", "2023-03-26 14:00:00"),
    ]

    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    for symbol, shares, price_per_share, total_price, transaction_type, transaction_date in dummy_transactions:
        c.execute("INSERT INTO stocks (symbol, shares, price_per_share, total_price, transaction_type, transaction_date) VALUES (?, ?, ?, ?, ?, ?)",
                  (symbol, shares, price_per_share, total_price, transaction_type, transaction_date))
    conn.commit()
    conn.close()

@app.route('/insert_dummy_data')
def insert_dummy_data_route():
    insert_dummy_data()
    flash('Dummy data inserted successfully', 'success')
    return redirect(url_for('index'))

@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction_route():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        shares = float(request.form['shares'])
        total_price = float(request.form['total_price'])
        transaction_type = request.form['transaction_type']
        currency = request.form['currency']
        
        if currency == 'SAR':
            total_price = total_price / 3.75  # Convert SAR to USD
        
        price_per_share = total_price / shares if shares > 0 else 0
        
        add_transaction(symbol, shares, price_per_share, total_price, transaction_type)
        flash(f"{transaction_type.capitalize()} transaction added successfully", 'success')
        return redirect(url_for('index'))
    return render_template('add_transaction.html')
def delete_transaction(transaction_id):
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("DELETE FROM stocks WHERE id=?", (transaction_id,))
    conn.commit()
    conn.close()

@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
def delete_transaction_route(transaction_id):
    delete_transaction(transaction_id)
    flash('Transaction deleted successfully', 'success')
    return redirect(url_for('transaction_history'))

@app.route('/transaction_history')
def transaction_history():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("SELECT id, symbol, shares, price_per_share, total_price, transaction_type, transaction_date FROM stocks ORDER BY transaction_date DESC")
    transactions = c.fetchall()
    conn.close()
    return render_template('transaction_history.html', transactions=transactions)

@app.route('/import_csv', methods=['GET', 'POST'])
def import_csv():
    if request.method == 'POST':
        csv_data = request.form.get('csv_data')
        if csv_data:
            csv_file = StringIO(csv_data)
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                symbol = row[0].upper()
                shares = float(row[1])
                total_price = float(row[2])/3.75
                transaction_type = row[3]
                
                # Check if date is provided, otherwise use current date
                transaction_date = row[4] if len(row) > 4 else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Parse the date to ensure it's in the correct format
                try:
                    parsed_date = parser.parse(transaction_date, fuzzy=True)
                except (ValueError, TypeError):
                    flash(f"Invalid date format for transaction: {symbol}, skipping...", 'danger')
                    continue

                # Format parsed_date as only date, without time, for yfinance compatibility
                formatted_date = parsed_date.strftime("%Y-%m-%d")

                price_per_share = total_price / (shares) if shares > 0 else 0

                # Manually insert the transaction with the full date and time
                conn = sqlite3.connect('portfolio.db')
                c = conn.cursor()
                c.execute("INSERT INTO stocks (symbol, shares, price_per_share, total_price, transaction_type, transaction_date) VALUES (?, ?, ?, ?, ?, ?)",
                          (symbol, shares, price_per_share, total_price, transaction_type, parsed_date.strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                conn.close()

            flash('CSV data imported successfully', 'success')
            return redirect(url_for('index'))
    return render_template('import_csv.html')

@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect('portfolio.db')
    c = conn.cursor()
    c.execute("SELECT symbol, shares, total_price, transaction_type, transaction_date FROM stocks")
    transactions = c.fetchall()
    conn.close()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Symbol', 'Shares', 'Total Price', 'Transaction Type', 'Transaction Date'])
    writer.writerows(transactions)

    return output.getvalue(), 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=portfolio_export.csv'}
@app.route('/portfolio_distribution')
def portfolio_distribution():
    portfolio_data, _, _, _, _ = calculate_portfolio_value("1d")
    df = pd.DataFrame(portfolio_data)
    fig = px.pie(df, values='market_value', names='symbol', title='Portfolio Distribution')
    graph_json = fig.to_json()
    return render_template('portfolio_distribution.html', graph_json=graph_json)

@app.route('/stock_performance')
def stock_performance():
    portfolio_data, _, _, _, _ = calculate_portfolio_value("1d")
    symbols = [stock['symbol'] for stock in portfolio_data]
    performances = []
    for symbol in symbols:
        data = yf.Ticker(symbol).history(period="1mo")
        performance = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
        performances.append({'symbol': symbol, 'performance': performance})
    
    df = pd.DataFrame(performances)
    fig = px.bar(df, x='symbol', y='performance', title='1-Month Stock Performance')
    graph_json = fig.to_json()
    return render_template('stock_performance.html', graph_json=graph_json)

@app.route('/rebalance', methods=['GET', 'POST'])
def rebalance():
    portfolio_data, total_value, _, _, _ = calculate_portfolio_value("1d")
    
    if request.method == 'POST':
        target_allocations = {}
        for stock in portfolio_data:
            target_allocations[stock['symbol']] = float(request.form.get(f"target_{stock['symbol']}", 0))
        
        rebalance_suggestions = []
        for stock in portfolio_data:
            symbol = stock['symbol']
            current_allocation = stock['market_value'] / total_value * 100
            target_allocation = target_allocations[symbol]
            difference = target_allocation - current_allocation
            
            if abs(difference) > 1:  # Only suggest changes for differences greater than 1%
                shares_to_change = round((difference / 100 * total_value) / stock['current_price'])
                action = "Buy" if difference > 0 else "Sell"
                rebalance_suggestions.append({
                    'symbol': symbol,
                    'action': action,
                    'shares': abs(shares_to_change),
                    'current_allocation': current_allocation,
                    'target_allocation': target_allocation
                })
        
        return render_template('rebalance.html', portfolio=portfolio_data, total_value=total_value, rebalance_suggestions=rebalance_suggestions)
    
    return render_template('rebalance.html', portfolio=portfolio_data, total_value=total_value)

@app.route('/generate_pdf_report')
def generate_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Portfolio Performance Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    portfolio_data, total_value, total_cost, total_gain_loss, _ = calculate_portfolio_value("1d")
    
    # Portfolio Summary
    elements.append(Paragraph("Portfolio Summary", styles['Heading2']))
    summary_data = [
        ["Total Value", f"${total_value:.2f}"],
        ["Total Cost", f"${total_cost:.2f}"],
        ["Total Gain/Loss", f"${total_gain_loss:.2f}"],
    ]
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))
    
    # Portfolio Composition
    elements.append(Paragraph("Portfolio Composition", styles['Heading2']))
    composition_data = [["Symbol", "Shares", "Current Price", "Market Value", "Cost Basis", "Gain/Loss"]]
    for stock in portfolio_data:
        composition_data.append([
            stock['symbol'],
            f"{stock['shares']:.2f}",
            f"${stock['current_price']:.2f}",
            f"${stock['market_value']:.2f}",
            f"${stock['cost']:.2f}",
            f"${stock['gain_loss']:.2f}"
        ])
    composition_table = Table(composition_data)
    composition_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(composition_table)
    elements.append(Spacer(1, 12))
    
    doc.build(elements)
    
    buffer.seek(0)
    return buffer.read(), 200, {'Content-Type': 'application/pdf', 'Content-Disposition': 'attachment; filename=portfolio_report.pdf'}

@app.route('/correlation_heatmap')
def correlation_heatmap():
    portfolio_data, _, _, _, _ = calculate_portfolio_value("1d")
    symbols = [stock['symbol'] for stock in portfolio_data]
    
    # Fetch historical data for all symbols
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Use 1 year of data
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Portfolio Correlation Heatmap')
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image to base64
    import base64
    heatmap_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return render_template('correlation_heatmap.html', heatmap_img=heatmap_img)

def is_market_open():
    # Define market hours (9:30 AM to 4:00 PM Eastern Time, Monday to Friday)
    now = datetime.now(pytz.timezone('US/Eastern'))
    market_start = datetime_time(9, 30)
    market_end = datetime_time(16, 0)
    print("Market is open: ", is_market_open())
    return (now.weekday() < 5 and
            market_start <= now.time() <= market_end)

def save_portfolio_value():
    if is_market_open():
        total_value = get_portfolio_value()
        conn = sqlite3.connect('portfolio.db')
        c = conn.cursor()
        c.execute("INSERT INTO hourly_values (timestamp, total_value) VALUES (?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), total_value))
        conn.commit()
        conn.close()
        print("Portfolio value saved: ", total_value)

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    create_database()
    schedule.every(10).seconds.do(save_portfolio_value)
    threading.Thread(target=run_schedule, daemon=True).start()
    app.run(debug=True, port=5000)
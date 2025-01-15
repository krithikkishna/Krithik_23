import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as tm
import Preprocess as pre
import RandomForest as RF
import DecisionTree as DT
import Hybrid as hy
import Predict as pr
import nltk
import DecisionTree
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction code here
    pass

if __name__ == '__main__':
    app.run(debug=True)
    from flask import Flask, request, render_template

    app = Flask(__name__)


    @app.route('/')
    def home():
        return render_template('index.html')


    @app.route('/predict', methods=['POST'])
    def predict():
        # Your prediction code here
        prediction_result = 'FAKE'  # Replace this with your actual prediction result
        return prediction_result


    if __name__ == '__main__':
        app.run(debug=True)

from flask import jsonify

@app.route('/predict')
def predict():
    # Your prediction logic here
    prediction_result = ...

    # Return a valid response object, such as jsonify() for JSON responses
    return jsonify(prediction_result)
from flask import Flask

# Instantiate the Flask application
app = Flask(__name__)

# Define routes and other configurations
@app.route('/')
def hello():
    return 'Hello, World!'

# Run the application
if __name__ == '__main__.py':
    app.run()

from flask import Flask, render_template, request

app = Flask(__name__)

nltk.download('wordnet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        message = request.form['message']

        # Perform fake news detection based on the input data
        prediction = perform_fake_news_detection(message)

        # Render the results template with the prediction
        return render_template('index.html', prediction=prediction)

def perform_fake_news_detection(message):
    # Your fake news detection logic goes here
    # This function should return the prediction ('FAKE' or 'REAL')
    return 'FAKE'  # Placeholder for demonstration purposes

bgcolor = "#DAF7A6"
bgcolor1 = "#B7C526"
fgcolor = "black"

def clear():
    print("Clear1")
    txt.delete(0, 'end')

def browse():
    path = filedialog.askopenfilename()  # Corrected function call
    print(path)
    txt.insert('end', path)
    if path != "":
        print(path)
    else:
        tm.showinfo("Input error", "Select Train Dataset")

def preprocess():
    try:
        pre.process()
        tm.showinfo("Input", "Preprocess Successfully Finished")
    except Exception as e:
        tm.showerror("Error", str(e))

def RFprocess():
    try:
        RF.process()
        tm.showinfo("Input", "RandomForest Successfully Finished")
    except Exception as e:
        tm.showerror("Error", str(e))

def DTprocess():
    try:
        DT.process()
        tm.showinfo("Input", "DT Successfully Finished")
    except Exception as e:
        tm.showerror("Error", str(e))

def hybridmodel():
    try:
        hy.process()
        tm.showinfo("Input", "Hybrid Successfully Finished")
    except Exception as e:
        tm.showerror("Error", str(e))

def predictprocess():
    sym = txt.get()
    if sym != "":
        try:
            res = pr.process(sym)
            tm.showinfo("Input", "Predicted Value is " + str(res))
        except Exception as e:
            tm.showerror("Error", str(e))
    else:
        tm.showinfo("Input error", "Give Input")

def logout():
    window.destroy()

window = tk.Tk()
# Create the Tkinter window object
window.title("Fake News")
window.geometry('1280x720')
window.configure(background=bgcolor)

message1 = tk.Label(window, text="Fake News", bg=bgcolor, fg=fgcolor, width=50, height=3,
                    font=('times', 30, 'italic bold underline'))
message1.place(x=100, y=20)

lbl = tk.Label(window, text="Input", width=20, height=2, fg=fgcolor, bg=bgcolor, font=('times', 15, ' bold '))
lbl.place(x=100, y=200)

txt = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
txt.place(x=400, y=215)

clearButton = tk.Button(window, text="Clear", command=clear, fg=fgcolor, bg=bgcolor1, width=20, height=2,
                        activebackground="Red", font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)

browseButton = tk.Button(window, text="Browse", command=browse, fg=fgcolor, bg=bgcolor1, width=20, height=2,
                         activebackground="Red", font=('times', 15, ' bold '))
browseButton.place(x=700, y=200)

process = tk.Button(window, text="Preprocess", command=preprocess, fg=fgcolor, bg=bgcolor1, width=17, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
process.place(x=10, y=600)

rfbutton = tk.Button(window, text="RANDOM FOREST", command=RFprocess, fg=fgcolor, bg=bgcolor1, width=17, height=2,
                     activebackground="Red", font=('times', 15, ' bold '))
rfbutton.place(x=210, y=600)

DTreebutton = tk.Button(window, text="Decision Tree", command=DTprocess, fg=fgcolor, bg=bgcolor1, width=17, height=2,
                        activebackground="Red", font=('times', 15, ' bold '))
DTreebutton.place(x=420, y=600)

NNbutton = tk.Button(window, text="Hybrid", command=hybridmodel, fg=fgcolor, bg=bgcolor1, width=17, height=2,
                     activebackground="Red", font=('times', 15, ' bold '))
NNbutton.place(x=620, y=600)

cbutton = tk.Button(window, text="Predict", command=predictprocess, fg=fgcolor, bg=bgcolor1, width=17, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
cbutton.place(x=820, y=600)

quitWindow = tk.Button(window, text="Quit", command=logout, fg=fgcolor, bg=bgcolor1, width=15, height=2,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=1020, y=600)

window.mainloop()

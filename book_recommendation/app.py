from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import pandas as pd
import joblib
import random

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

book_recommendations = {
    "Fantasy": ["The Hobbit by J.R.R. Tolkien", "A Game of Thrones by George R.R. Martin"],
    "Romance": ["Pride and Prejudice by Jane Austen", "Jane Eyre by Charlotte Bronte"],
    "Science Fiction": ["The Martian by Andy Weir", "Dune by Frank Herbert"],
    "Mystery": ["The Girl with the Dragon Tattoo by Stieg Larsson", "Gone Girl by Gillian Flynn"],
    "Historical Fiction": ["The Book Thief by Markus Zusak", "All the Light We Cannot See by Anthony Doerr"],
    "Adventure": ["Into the Wild by Jon Krakauer", "Life of Pi by Yann Martel"],
    "History": ["Sapiens by Yuval Noah Harari", "Guns, Germs, and Steel by Jared Diamond"],
    "Poetry": ["Leaves of Grass by Walt Whitman", "The Waste Land by T.S. Eliot"],
    "Thriller": ["The Da Vinci Code by Dan Brown", "Shutter Island by Dennis Lehane"],
    "Biography": ["Steve Jobs by Walter Isaacson", "The Diary of a Young Girl by Anne Frank"],
    "Self-Help": ["How to Win Friends and Influence People by Dale Carnegie", "Atomic Habits by James Clear"],
    "Science": ["A Brief History of Time by Stephen Hawking", "The Selfish Gene by Richard Dawkins"],
    "Astronomy": ["Cosmos by Carl Sagan", "Pale Blue Dot by Carl Sagan"],
    "Classic": ["To Kill a Mockingbird by Harper Lee", "1984 by George Orwell"],
    "Young Adult": ["The Hunger Games by Suzanne Collins", "Harry Potter and the Sorcerer's Stone by J.K. Rowling"],
    "Fiction": ["The Alchemist by Paulo Coelho", "The Great Gatsby by F. Scott Fitzgerald"],
    "Espionage": ["Tinker Tailor Soldier Spy by John le Carré", "The Spy Who Came in from the Cold by John le Carré"],
    "Art": ["The Story of Art by E.H. Gombrich", "Ways of Seeing by John Berger"],
    "Cooking": ["Mastering the Art of French Cooking by Julia Child", "The Joy of Cooking by Irma S. Rombauer"],
    "Drama": ["Death of a Salesman by Arthur Miller", "Hamlet by William Shakespeare"],
    "Film": ["Making Movies by Sidney Lumet", "Sculpting in Time by Andrei Tarkovsky"],
    "Music": ["Life by Keith Richards", "Chronicles, Vol. 1 by Bob Dylan"],
    "Philosophy": ["The Republic by Plato", "Meditations by Marcus Aurelius"],
    "Politics": ["The Prince by Niccolò Machiavelli", "1984 by George Orwell"],
    "Travel": ["In Patagonia by Bruce Chatwin", "Vagabonding by Rolf Potts"],
    "Business": ["The Lean Startup by Eric Ries", "Good to Great by Jim Collins"],
    "Environment": ["Silent Spring by Rachel Carson", "The Omnivore's Dilemma by Michael Pollan"],
    "Health": ["The China Study by T. Colin Campbell", "Why We Sleep by Matthew Walker"],
    "Memoir": ["Educated by Tara Westover", "Becoming by Michelle Obama"],
    "Mythology": ["The Power of Myth by Joseph Campbell", "Mythology by Edith Hamilton"],
    "Economics": ["Capital in the Twenty-First Century by Thomas Piketty", "The Wealth of Nations by Adam Smith"],
    "Religion": ["The God Delusion by Richard Dawkins", "The Power of Now by Eckhart Tolle"],
    "Literature": ["Moby-Dick by Herman Melville", "War and Peace by Leo Tolstoy"],
    "Technology": ["The Innovators by Walter Isaacson", "Hooked by Nir Eyal"]
}


@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/clear-session')
def clear_session():
    session.clear()
    return redirect(url_for('home'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'conversation' not in session:
        session['conversation'] = []

    user_input = request.form['message']
    transformed_input = vectorizer.transform([user_input])
    genre_prediction = model.predict(transformed_input)[0]
    print(genre_prediction)

    book_suggestions = book_recommendations.get(genre_prediction)

    if book_suggestions:
        random_book = random.choice(book_suggestions)
        response = f"I recommend this book: {random_book}"
    else:
        # Default message if no books are available for the genre or if the genre is not found
        response = "Sorry, I don't have any recommendations for that genre."

    session['conversation'].append(("user", user_input))
    session['conversation'].append(("bot", response))
    session.modified = True 

    message_count = len(session['conversation']) // 2

    return render_template('index.html', conversation=session['conversation'], message_count=message_count)

if __name__ == '__main__':
    app.run(debug=True)

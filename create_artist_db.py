import sqlite3

def create_and_seed_simple_db():
    # Crée ou écrase le fichier SQLite local
    conn = sqlite3.connect("artists_genres.db")
    cursor = conn.cursor()

    # Suppression des anciennes tables si elles existaient pour repartir sur du propre
    cursor.execute("DROP TABLE IF EXISTS artist_genres;")
    cursor.execute("DROP TABLE IF EXISTS artists;")

    # Création de la table unique à deux colonnes comme demandé
    cursor.execute("""
    CREATE TABLE artists (
        artist TEXT NOT NULL UNIQUE,
        genre TEXT NOT NULL
    );
    """)

    # Indexation pour des performances de recherche maximales
    cursor.execute("CREATE INDEX idx_artist_name ON artists(artist);")

    # Dictionnaire propre des 200 artistes avec leur genre principal unique
    data = {
        "Pop": [
            "Taylor Swift", "Billie Eilish", "Olivia Rodrigo", "Sabrina Carpenter", "Dua Lipa",
            "Ariana Grande", "Chappell Roan", "The Weeknd", "Bruno Mars", "Ed Sheeran",
            "Justin Bieber", "Tate McRae", "Charli XCX", "Miley Cyrus", "Harry Styles",
            "Selena Gomez", "Shawn Mendes", "Camila Cabello", "Katy Perry", "Lady Gaga",
            "Pink", "Troye Sivan", "Gracie Abrams", "Benson Boone", "Teddy Swims",
            "Renee Rapp", "LANY", "Lauv", "Lorde", "Halsey", "Kelly Clarkson",
            "Charlie Puth", "Meghan Trainor", "Niall Horan", "Louis Tomlinson", "Sam Smith",
            "Khalid", "Conan Gray", "Melanie Martinez", "Fletcher", "Clairo",
            "Madison Beer", "Tove Lo", "Kim Petras", "Remi Wolf", "Maggie Rogers",
            "Caroline Polachek", "Nelly Furtado", "Gwen Stefani", "Maroon 5"
        ],
        "Rap": [
            "Drake", "Kendrick Lamar", "Travis Scott", "Kanye West (Ye)", "Eminem",
            "Nicki Minaj", "Post Malone", "Megan Thee Stallion", "Cardi B", "SZA",
            "Future", "Lil Baby", "21 Savage", "Jack Harlow", "Playboi Carti",
            "Gunna", "Lil Nas X", "Ice Spice", "Doja Cat", "Latto", "GloRilla",
            "Central Cee", "Don Toliver", "Chris Brown", "Usher", "Summer Walker",
            "Kehlani", "Brent Faiyaz", "J. Cole", "ASAP Rocky"
        ],
        "Latin": [
            "Bad Bunny", "Peso Pluma", "Karol G", "Feid", "Rauw Alejandro",
            "J Balvin", "Maluma", "Shakira", "Rosalía", "Daddy Yankee",
            "Myke Towers", "El Alfa", "Fuerza Regida", "Grupo Frontera", "Carin León",
            "Natanael Cano", "Christian Nodal", "Anuel AA", "Ozuna", "Manuel Turizo",
            "Camilo", "Sebastián Yatra", "Becky G", "Luis Miguel", "Alejandro Fernández"
        ],
        "Electro": [
            "Illenium", "Fred again..", "Skrillex", "Tiësto", "Calvin Harris",
            "Martin Garrix", "David Guetta", "Marshmello", "Steve Aoki", "Zedd",
            "Kygo", "DJ Snake", "Kaskade", "Deadmau5", "Excision", "Subtronics",
            "Flume", "ODESZA", "Rufus Du Sol", "Disclosure", "Chainsmokers",
            "Alesso", "Afrojack", "Gryffin", "San Holo"
        ],
        "Techno": [
            "John Summit", "Dom Dolla", "Peggy Gou", "Fisher", "Charlotte de Witte",
            "Amelie Lens", "Carl Cox", "Boris Brejcha", "Chris Lake", "Michael Bibi",
            "Pawsa", "Vintage Culture", "Mochakk", "Green Velvet", "Nicole Moudaber"
        ],
        "Rock": [
            "Foo Fighters", "blink-182", "Green Day", "Red Hot Chili Peppers", "Fall Out Boy",
            "Paramore", "Coldplay", "Imagine Dragons", "The Killers", "Arctic Monkeys",
            "Cage the Elephant", "My Chemical Romance", "Weezer", "Gorillaz", "Muse",
            "Kings of Leon", "Queens of the Stone Age", "Greta Van Fleet", "Thirty Seconds to Mars", "Shinedown",
            "Bring Me The Horizon", "The 1975", "Bleachers", "Pierce the Veil", "Deftones",
            "Incubus", "Jimmy Eat World", "All Time Low", "The Smashing Pumpkins", "Third Eye Blind"
        ],
        "Country": [
            "Morgan Wallen", "Luke Combs", "Zach Bryan", "Lainey Wilson", "Chris Stapleton",
            "Jelly Roll", "Kane Brown", "Cody Johnson", "Kacey Musgraves", "Thomas Rhett"
        ],
        "Blues": [
            "Leon Bridges", "Gary Clark Jr.", "Jon Batiste", "Norah Jones", "John Mayer",
            "Hozier", "Brittany Howard", "Marcus King", "Michael Kiwanuka", "Trombone Shorty"
        ],
        "Classical": [
            "New York Philharmonic", "Los Angeles Philharmonic (LA Phil)", "Yo-Yo Ma", "Lang Lang", "Itzhak Perlman",
            "Chicago Symphony Orchestra", "Boston Symphony Orchestra", "Andrea Bocelli", "Hilary Hahn", "Gustavo Dudamel"
        ],
        "Metal": [
            "Metallica", "Slipknot", "Tool", "Avenged Sevenfold", "Korn",
            "Disturbed", "Iron Maiden", "System of a Down", "Ghost"
        ],
        "Reggae": [
            "Burna Boy", "Stick Figure", "Rebelution", "Damian Marley", "Slightly Stoopid",
            "Sean Paul", "Shaggy", "Iration", "Chronixx", "Stephen Marley"
        ]
    }

    # Insertion en masse dans la table
    print("⏳ Génération de la base de données simplifiée...")
    for genre_name, artist_list in data.items():
        for artist_name in artist_list:
            try:
                cursor.execute(
                    "INSERT INTO artists (artist, genre) VALUES (?, ?)",
                    (artist_name, genre_name)
                )
            except sqlite3.IntegrityError:
                pass

    conn.commit()

    # Petit print de vérification
    cursor.execute("SELECT COUNT(*) FROM artists;")
    total = cursor.fetchone()[0]
    conn.close()
    

if __name__ == "__main__":
    create_and_seed_simple_db()
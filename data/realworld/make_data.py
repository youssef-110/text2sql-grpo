"""Generate the four real-world eval databases as SQL text scripts. Stdlib only, fixed seeds.

Run: python data/realworld/make_data.py
All people, clubs and leagues are invented.
"""
import random
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent

FIRST = ("Amira Omar Lina Karim Sara Youssef Nora Hassan Maya Tarek Emma Liam Olivia Noah Ava Lucas Mia Ethan "
         "Sofia Mateo Chloe Leo Hannah Felix Clara Jonas Elena Marco Giulia Pablo Ines Diego Priya Arjun Mei Kenji").split()
LAST = ("Haddad Mansour Farouk Nasser Salem Rahman Whitfield Carver Ellison Thorne Hartley Mercer Vogel Brandt "
        "Keller Moreau Laurent Dubois Rossi Bianchi Conti Navarro Ortega Vega Sharma Iyer Tanaka Sato Novak Petrov").split()


def people(n):
    """n unique full names."""
    return random.sample([f"{f} {l}" for f in FIRST for l in LAST], n)


def day(a, b):
    return a + timedelta(days=random.randint(0, (b - a).days))


def lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, date):
        v = v.isoformat()
    return "'" + v.replace("'", "''") + "'" if isinstance(v, str) else repr(v)


def write(name, schema, tables, main):
    for t in main:
        assert 30 <= len(tables[t]) <= 200, (t, len(tables[t]))
    lines = [schema.strip(), ""]
    for table, rows in tables.items():
        lines += [f"INSERT INTO {table} VALUES ({', '.join(map(lit, r))});" for r in rows]
        lines.append("")
    (HERE / f"{name}.sql").write_text("\n".join(lines), encoding="utf-8", newline="\n")


FOOTBALL_SCHEMA = """
CREATE TABLE leagues (
    league_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    tier INTEGER NOT NULL
);
CREATE TABLE clubs (
    club_id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL REFERENCES leagues(league_id),
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    founded_year INTEGER NOT NULL,
    stadium TEXT NOT NULL,
    capacity INTEGER NOT NULL
);
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    club_id INTEGER NOT NULL REFERENCES clubs(club_id),
    name TEXT NOT NULL,
    position TEXT NOT NULL CHECK (position IN ('GK', 'DEF', 'MID', 'FWD')),
    nationality TEXT NOT NULL,
    birth_date TEXT NOT NULL,
    shirt_number INTEGER NOT NULL,
    market_value_eur INTEGER
);
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY,
    league_id INTEGER NOT NULL REFERENCES leagues(league_id),
    season TEXT NOT NULL,
    match_date TEXT NOT NULL,
    home_club_id INTEGER NOT NULL REFERENCES clubs(club_id),
    away_club_id INTEGER NOT NULL REFERENCES clubs(club_id),
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    attendance INTEGER NOT NULL
);
CREATE TABLE goals (
    goal_id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    club_id INTEGER NOT NULL REFERENCES clubs(club_id),
    minute INTEGER NOT NULL,
    is_penalty INTEGER NOT NULL DEFAULT 0
);
"""


def football():
    random.seed(101)
    leagues = [(1, "Albion Premier League", "England", 1), (2, "Albion Championship", "England", 2),
               (3, "Liga Meridional", "Spain", 1), (4, "Serie Adriatica", "Italy", 1)]
    towns = {
        1: ["Kingsbridge Rovers", "Ashbourne United", "Marlow Athletic", "Thornbury City",
            "Wexcombe Town", "Brackley Albion", "Holmfield Rangers", "Dunmere Wanderers"],
        2: ["Fenwick Town", "Oakhampton City", "Redcliffe United", "Silverdale Rovers",
            "Tarnwick Athletic", "Winterbourne Albion", "Elmsworth Rangers", "Larkfield Wanderers"],
        3: ["Valdoro CF", "Montalba CF", "Riosur Deportivo", "Sierranova CF",
            "Puertoclaro Atletico", "Aguaviva Deportivo", "Castelmar CF", "Lomaverde Atletico"],
        4: ["Montelupo Calcio", "Portoverde Calcio", "Castelrosso Sporting", "Valtorre Calcio",
            "Rocca Azzurra Sporting", "Fontebella Calcio", "Sanmarco Vecchio Sporting", "Lagonero Calcio"],
    }
    clubs, by_league = [], {}
    for lid, names in towns.items():
        for name in names:
            cid = len(clubs) + 1
            city = name.rsplit(" ", 1)[0]
            clubs.append((cid, lid, name, city, random.randint(1880, 1965),
                          f"{city} {random.choice(['Park', 'Arena', 'Stadium', 'Ground'])}",
                          random.randrange(8000, 60000, 500)))
            by_league.setdefault(lid, []).append(cid)
    capacity = {c[0]: c[6] for c in clubs}

    players, squad = [], {}
    names = iter(people(192))
    nations = ["England", "Spain", "Italy", "France", "Brazil", "Egypt", "Germany", "Nigeria"]
    for club in clubs:
        for pos, shirt in zip(["GK", "DEF", "DEF", "MID", "FWD", "FWD"], random.sample(range(1, 31), 6)):
            pid = len(players) + 1
            value = None if random.random() < 0.06 else random.randrange(150_000, 45_000_000, 50_000)
            players.append((pid, club[0], next(names), pos, random.choice(nations),
                            day(date(1992, 1, 1), date(2006, 12, 31)), shirt, value))
            squad.setdefault(club[0], []).append((pid, pos))

    fixtures = []
    seasons = [("2023-24", date(2023, 8, 12), date(2024, 5, 19)), ("2024-25", date(2024, 8, 10), date(2025, 5, 25))]
    for lid, ids in by_league.items():
        for season, start, end in seasons:
            for _ in range(10):
                home, away = random.sample(ids, 2)
                fixtures.append((day(start, end), lid, season, home, away))
    fixtures.sort()

    matches, goals = [], []
    budget = 200  # the goals table must stay within the plan's row cap and always equal the scorelines
    weights = {"GK": 0, "DEF": 1, "MID": 3, "FWD": 6}
    for mid, (when, lid, season, home, away) in enumerate(fixtures, 1):
        hs = min(budget, random.choices(range(5), [24, 34, 26, 11, 5])[0])
        budget -= hs
        aws = min(budget, random.choices(range(4), [36, 36, 20, 8])[0])
        budget -= aws
        matches.append((mid, lid, season, when, home, away, hs, aws,
                        random.randint(capacity[home] * 2 // 5, capacity[home])))
        sides = [home] * hs + [away] * aws
        for club, minute in zip(random.sample(sides, len(sides)), sorted(random.sample(range(1, 91), len(sides)))):
            pool = squad[club]
            pid = random.choices([p for p, _ in pool], [weights[pos] for _, pos in pool])[0]
            goals.append((len(goals) + 1, mid, pid, club, minute, int(random.random() < 0.12)))
    write("football", FOOTBALL_SCHEMA,
          {"leagues": leagues, "clubs": clubs, "players": players, "matches": matches, "goals": goals},
          ["clubs", "players", "matches", "goals"])


SHOP_SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL,
    discontinued INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'shipped', 'delivered', 'cancelled', 'returned')),
    shipping_cost REAL NOT NULL
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount REAL NOT NULL DEFAULT 0
);
"""


def shop():
    random.seed(202)
    places = [("Cairo", "Egypt"), ("Alexandria", "Egypt"), ("Dubai", "UAE"), ("Riyadh", "Saudi Arabia"),
              ("London", "UK"), ("Manchester", "UK"), ("Berlin", "Germany"), ("Paris", "France"),
              ("Toronto", "Canada"), ("Chicago", "USA")]
    customers = []
    for cid, name in enumerate(people(60), 1):
        city, country = random.choice(places)
        phone = None if random.random() < 0.2 else f"+1-555-{random.randint(1000, 9999)}"
        customers.append((cid, name, name.lower().replace(" ", ".") + "@example.com", phone, city, country,
                          day(date(2022, 1, 1), date(2023, 12, 31))))
    catalog = {
        "Electronics": [("Wireless Earbuds", 59.99), ("USB-C Hub", 34.5), ("Mechanical Keyboard", 89.0), ("4K Monitor", 329.0),
                        ("Webcam HD", 49.95), ("Portable SSD 1TB", 109.0), ("Smart Speaker", 79.0), ("Laptop Stand", 29.99)],
        "Home": [("Ceramic Mug Set", 24.0), ("Bamboo Cutting Board", 18.5), ("LED Desk Lamp", 39.0), ("Linen Curtains", 64.0),
                 ("Cast Iron Skillet", 44.95), ("Scented Candle", 12.5), ("Throw Blanket", 35.0), ("Storage Baskets", 27.0)],
        "Sports": [("Yoga Mat", 25.0), ("Running Belt", 15.99), ("Adjustable Dumbbells", 149.0), ("Resistance Bands", 19.5),
                   ("Cycling Gloves", 22.0), ("Water Bottle", 14.0), ("Foam Roller", 28.5), ("Tennis Racket", 119.0)],
        "Books": [("Intro to Databases", 42.0), ("The Quiet Harbor", 16.0), ("Cooking for Two", 23.5), ("A Short History of Maps", 19.0),
                  ("Learning Python", 54.0), ("Gardening Basics", 17.5), ("The Last Lighthouse", 14.5), ("Travel Sketchbook", 21.0)],
        "Toys": [("Wooden Blocks", 26.0), ("Puzzle 1000 Pieces", 18.0), ("Remote Control Car", 58.0), ("Plush Fox", 16.5),
                 ("Board Game Night", 37.0), ("Kite Flyer", 13.0), ("Marble Run", 33.0), ("Science Kit", 45.0)],
    }
    products = []
    for category, items in catalog.items():
        for name, price in items:
            products.append((len(products) + 1, name, category, price,
                             0 if random.random() < 0.1 else random.randint(3, 300), int(random.random() < 0.1)))
    random.shuffle(products)
    products = [(i, *p[1:]) for i, p in enumerate(products, 1)]
    price = {p[0]: p[3] for p in products}

    # The last few customers and products never appear, so "never ordered" questions have answers.
    raw = []
    for _ in range(90):
        cust = random.choice(customers[:52])
        raw.append((day(max(cust[6], date(2023, 1, 3)), date(2024, 12, 28)), cust[0]))
    raw.sort()
    statuses = ["delivered", "shipped", "processing", "cancelled", "returned"]
    orders, items = [], []
    for oid, (when, cid) in enumerate(raw, 1):
        orders.append((oid, cid, when, random.choices(statuses, [62, 10, 6, 14, 8])[0],
                       random.choice([0.0, 4.99, 7.5, 12.0])))
        for pid in random.sample(range(1, 37), random.randint(1, 3)):
            items.append((len(items) + 1, oid, pid, random.choices([1, 2, 3, 4], [55, 28, 12, 5])[0], price[pid],
                          random.choices([0.0, 0.1, 0.2], [75, 18, 7])[0]))
    write("shop", SHOP_SCHEMA, {"customers": customers, "products": products, "orders": orders, "order_items": items},
          ["customers", "products", "orders", "order_items"])


SCHOOL_SCHEMA = """
CREATE TABLE teachers (
    teacher_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    hire_date TEXT NOT NULL,
    office TEXT
);
CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    department TEXT NOT NULL,
    credits INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL REFERENCES teachers(teacher_id)
);
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    birth_date TEXT NOT NULL,
    enrollment_year INTEGER NOT NULL,
    major TEXT NOT NULL
);
CREATE TABLE enrollments (
    enrollment_id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(student_id),
    course_id INTEGER NOT NULL REFERENCES courses(course_id),
    term TEXT NOT NULL,
    enrolled_on TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'in progress', 'dropped'))
);
CREATE TABLE grades (
    grade_id INTEGER PRIMARY KEY,
    enrollment_id INTEGER NOT NULL REFERENCES enrollments(enrollment_id),
    assessment TEXT NOT NULL CHECK (assessment IN ('midterm', 'final')),
    score REAL NOT NULL,
    graded_on TEXT NOT NULL
);
"""


def school():
    random.seed(303)
    titles = {
        "Mathematics": ["Calculus I", "Calculus II", "Linear Algebra", "Discrete Mathematics", "Probability", "Number Theory"],
        "Computer Science": ["Intro to Programming", "Data Structures", "Databases", "Operating Systems", "Algorithms", "Computer Networks"],
        "Physics": ["Mechanics", "Electromagnetism", "Thermodynamics", "Optics", "Quantum Physics", "Astrophysics"],
        "History": ["World History", "Ancient Civilizations", "Medieval Europe", "Modern Middle East", "History of Science", "Cold War Studies"],
        "English": ["Academic Writing", "Shakespeare", "Modern Poetry", "Creative Writing", "The Novel", "Rhetoric"],
        "Biology": ["Cell Biology", "Genetics", "Ecology", "Human Anatomy", "Microbiology", "Evolution"],
    }
    depts = list(titles)
    teachers = []
    for tid, name in enumerate(people(30), 1):
        office = None if random.random() < 0.1 else f"{random.choice('ABCD')}-{random.randint(101, 420)}"
        teachers.append((tid, name, depts[(tid - 1) % 6], day(date(2005, 1, 1), date(2022, 8, 31)), office))
    staff = {d: [t[0] for t in teachers if t[2] == d] for d in depts}
    prefix = {"Mathematics": "MATH", "Computer Science": "CS", "Physics": "PHYS", "History": "HIST", "English": "ENG", "Biology": "BIO"}
    courses = []
    for d in depts:
        for i, title in enumerate(titles[d], 1):
            courses.append((len(courses) + 1, f"{prefix[d]}{100 * (1 + (i - 1) // 2) + i}", title, d,
                            random.choice([2, 3, 3, 4]), random.choice(staff[d])))
    random.shuffle(courses)
    courses = [(i, *c[1:]) for i, c in enumerate(courses, 1)]
    students = [(sid, name, day(date(2002, 1, 1), date(2006, 12, 31)), random.randint(2021, 2024), random.choice(depts))
                for sid, name in enumerate(people(80), 1)]
    ability = {s[0]: random.gauss(74, 8) for s in students}

    terms = {"Fall 2023": date(2023, 9, 4), "Spring 2024": date(2024, 2, 5), "Fall 2024": date(2024, 9, 2)}
    # The last students and courses stay empty, so "never enrolled" questions have answers.
    pairs = random.sample([(s, c) for s in range(1, 73) for c in range(1, 34)], 120)
    enrollments, grades = [], []
    for eid, (sid, cid) in enumerate(pairs, 1):
        term = random.choice(list(terms))
        start = terms[term]
        if random.random() < 0.1:
            status = "dropped"
        else:
            status = "in progress" if term == "Fall 2024" else "completed"
        enrollments.append((eid, sid, cid, term, start - timedelta(days=random.randint(1, 20)), status))
        if status == "dropped":
            continue
        for kind, week in [("midterm", 7), ("final", 15)][:1 if status == "in progress" else 2]:
            score = round(min(100.0, max(20.0, random.gauss(ability[sid], 9))), 1)
            grades.append((len(grades) + 1, eid, kind, score, start + timedelta(weeks=week, days=random.randint(0, 4))))
    write("school", SCHOOL_SCHEMA,
          {"teachers": teachers, "courses": courses, "students": students, "enrollments": enrollments, "grades": grades},
          ["teachers", "courses", "students", "enrollments", "grades"])


HR_SCHEMA = """
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT NOT NULL,
    annual_budget INTEGER NOT NULL
);
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    job_title TEXT NOT NULL,
    department_id INTEGER NOT NULL REFERENCES departments(department_id),
    manager_id INTEGER REFERENCES employees(employee_id),
    hire_date TEXT NOT NULL,
    termination_date TEXT
);
CREATE TABLE salaries (
    salary_id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL REFERENCES employees(employee_id),
    amount INTEGER NOT NULL,
    effective_date TEXT NOT NULL,
    end_date TEXT
);
"""


def hr():
    random.seed(404)
    roles = {
        "Engineering": ["Software Engineer", "QA Engineer", "DevOps Engineer"],
        "Sales": ["Account Executive", "Sales Representative"],
        "Marketing": ["Marketing Specialist", "Content Writer"],
        "Finance": ["Accountant", "Financial Analyst"],
        "Human Resources": ["Recruiter", "HR Coordinator"],
        "Operations": ["Operations Analyst", "Logistics Coordinator"],
        "Customer Support": ["Support Agent", "Support Specialist"],
    }
    places = ["Cairo", "London", "Berlin", "Toronto", "Dubai", "Paris", "Cairo", "London"]
    budgets = [900_000, 4_800_000, 2_600_000, 1_400_000, 1_100_000, 900_000, 1_700_000, 1_300_000]
    dept_names = ["Executive", *roles]
    departments = [(i, n, places[i - 1], budgets[i - 1]) for i, n in enumerate(dept_names, 1)]
    dept_id = {n: i for i, n in enumerate(dept_names, 1)}
    names = people(80)

    def person(eid, title, dept, manager, hired, band):
        name = names[eid - 1]
        left = None
        if band == "staff" and random.random() < 0.12:
            left = day(max(hired + timedelta(days=200), date(2023, 1, 1)), date(2024, 12, 15))
        return (eid, name, name.lower().replace(" ", ".") + "@example.com", title, dept_id[dept], manager, hired, left)

    staff = [person(1, "Chief Executive Officer", "Executive", None, date(2012, 3, 1), "ceo")]
    base = {1: 240_000}
    heads, leads = {}, {}
    for eid, dept in enumerate(roles, 2):
        heads[dept] = eid
        staff.append(person(eid, f"Head of {dept}", dept, 1, day(date(2013, 1, 1), date(2017, 12, 31)), "head"))
        base[eid] = random.randrange(140_000, 182_000, 500)
    depts = list(roles)
    for eid in range(9, 21):
        dept = depts[(eid - 9) % len(depts)]
        leads.setdefault(dept, []).append(eid)
        staff.append(person(eid, "Team Lead", dept, heads[dept], day(date(2015, 1, 1), date(2020, 12, 31)), "lead"))
        base[eid] = random.randrange(95_000, 132_000, 500)
    for eid in range(21, 81):
        dept = random.choices(depts, [30, 16, 9, 9, 6, 12, 18])[0]
        manager = random.choice(leads.get(dept) or [heads[dept]])
        staff.append(person(eid, random.choice(roles[dept]), dept, manager, day(date(2016, 1, 1), date(2024, 6, 30)), "staff"))
        base[eid] = random.randrange(48_000, 92_000, 500)

    extras = list(range(1, 81))
    random.shuffle(extras)
    raises = {e: 0 for e in extras}
    for e in (extras * 2)[:120]:
        raises[e] += 1
    salaries = []
    for emp in staff:
        eid, hired, left = emp[0], emp[6], emp[7]
        last_day = left or date(2024, 12, 31)
        dates = [hired]
        for _ in range(raises[eid]):
            nxt = dates[-1] + timedelta(days=random.randint(330, 400))
            if nxt < last_day:
                dates.append(nxt)
        amount = base[eid]
        for i, start in enumerate(dates):
            end = dates[i + 1] - timedelta(days=1) if i + 1 < len(dates) else left
            salaries.append((len(salaries) + 1, eid, amount, start, end))
            amount = int(round(amount * random.uniform(1.03, 1.12), -2))
    write("hr", HR_SCHEMA, {"departments": departments, "employees": staff, "salaries": salaries}, ["employees", "salaries"])


if __name__ == "__main__":
    football()
    shop()
    school()
    hr()

import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


TRAINING_DATA = [
    # Food & Dining
    ("big mac meal", "food_dining"),
    ("quarter pounder", "food_dining"),
    ("mcflurry oreo", "food_dining"),
    ("burger king whopper", "food_dining"),
    ("pizza margherita", "food_dining"),
    ("pepperoni pizza", "food_dining"),
    ("coffee latte", "food_dining"),
    ("espresso double", "food_dining"),
    ("cappuccino", "food_dining"),
    ("smoothie mango", "food_dining"),
    ("chicken wings", "food_dining"),
    ("sushi rolls", "food_dining"),
    ("pad thai noodles", "food_dining"),
    ("caesar salad", "food_dining"),
    ("club sandwich", "food_dining"),
    ("french fries large", "food_dining"),
    ("cola 500ml", "food_dining"),
    ("iced tea", "food_dining"),
    ("croissant", "food_dining"),
    ("blueberry muffin", "food_dining"),
    ("breakfast combo", "food_dining"),
    ("lunch special", "food_dining"),
    ("dinner set", "food_dining"),
    ("happy meal", "food_dining"),
    ("apple pie slice", "food_dining"),

    # Groceries
    ("whole milk 1 gallon", "groceries"),
    ("skim milk 2l", "groceries"),
    ("bread wheat 20oz", "groceries"),
    ("sourdough bread", "groceries"),
    ("chicken breast boneless", "groceries"),
    ("ground beef 1lb", "groceries"),
    ("salmon fillet", "groceries"),
    ("eggs large 12 count", "groceries"),
    ("greek yogurt 32oz", "groceries"),
    ("cheddar cheese block", "groceries"),
    ("butter unsalted", "groceries"),
    ("orange juice 1.75l", "groceries"),
    ("apple juice 1l", "groceries"),
    ("mineral water 6pack", "groceries"),
    ("pasta spaghetti 500g", "groceries"),
    ("tomato sauce 24oz", "groceries"),
    ("olive oil 500ml", "groceries"),
    ("rice basmati 5kg", "groceries"),
    ("wheat flour 1kg", "groceries"),
    ("sugar 1kg", "groceries"),
    ("salt iodized", "groceries"),
    ("onion 1kg", "groceries"),
    ("tomato 500g", "groceries"),
    ("potato bag 5lb", "groceries"),
    ("banana bunch", "groceries"),
    ("apple red 6pcs", "groceries"),
    ("orange bag", "groceries"),
    ("cereal cornflakes", "groceries"),
    ("oats rolled 1kg", "groceries"),
    ("jam strawberry", "groceries"),
    ("peanut butter", "groceries"),
    ("honey 500g", "groceries"),
    ("chips lays", "groceries"),
    ("biscuits digestive", "groceries"),
    ("snack bar protein", "groceries"),

    # Transport
    ("uber trip downtown", "transport"),
    ("lyft ride airport", "transport"),
    ("ola auto ride", "transport"),
    ("rapido bike taxi", "transport"),
    ("base fare trip", "transport"),
    ("surge pricing", "transport"),
    ("booking fee uber", "transport"),
    ("metro card recharge", "transport"),
    ("bus ticket", "transport"),
    ("train ticket", "transport"),
    ("subway single journey", "transport"),
    ("taxi fare", "transport"),
    ("cab hire", "transport"),
    ("auto rickshaw", "transport"),
    ("shuttle service", "transport"),
    ("ferry ticket", "transport"),

    # Fuel
    ("petrol 15 litres", "fuel"),
    ("diesel refuel", "fuel"),
    ("fuel regular 87", "fuel"),
    ("gasoline fill up", "fuel"),
    ("cng refill", "fuel"),
    ("shell petrol", "fuel"),
    ("hpcl pump", "fuel"),
    ("iocl fuel station", "fuel"),
    ("bharat petroleum", "fuel"),

    # Healthcare
    ("doctor consultation", "healthcare"),
    ("physician visit", "healthcare"),
    ("hospital bill", "healthcare"),
    ("dental checkup", "healthcare"),
    ("dentist procedure", "healthcare"),
    ("blood test", "healthcare"),
    ("lab report", "healthcare"),
    ("xray chest", "healthcare"),
    ("mri scan", "healthcare"),
    ("ct scan", "healthcare"),
    ("eye checkup", "healthcare"),
    ("optician visit", "healthcare"),
    ("physiotherapy session", "healthcare"),
    ("clinic visit", "healthcare"),
    ("specialist consultation", "healthcare"),

    # Pharmacy
    ("paracetamol 500mg", "pharmacy"),
    ("ibuprofen tablets", "pharmacy"),
    ("vitamin c supplements", "pharmacy"),
    ("multivitamin capsules", "pharmacy"),
    ("antibiotic course", "pharmacy"),
    ("cough syrup", "pharmacy"),
    ("eye drops", "pharmacy"),
    ("antiseptic cream", "pharmacy"),
    ("bandage roll", "pharmacy"),
    ("first aid kit", "pharmacy"),
    ("pharmacy prescription", "pharmacy"),
    ("medplus medicines", "pharmacy"),
    ("apollo pharmacy", "pharmacy"),
    ("netmeds order", "pharmacy"),

    # Entertainment
    ("movie ticket imax", "entertainment"),
    ("cinema show", "entertainment"),
    ("concert ticket", "entertainment"),
    ("netflix subscription", "entertainment"),
    ("spotify premium", "entertainment"),
    ("amazon prime", "entertainment"),
    ("hotstar subscription", "entertainment"),
    ("steam game purchase", "entertainment"),
    ("ps5 game", "entertainment"),
    ("xbox game pass", "entertainment"),
    ("app store purchase", "entertainment"),
    ("bowling game", "entertainment"),
    ("laser tag", "entertainment"),
    ("amusement park ticket", "entertainment"),
    ("museum entry", "entertainment"),
    ("zoo ticket", "entertainment"),

    # Clothing
    ("tshirt cotton", "clothing"),
    ("denim jeans", "clothing"),
    ("formal shirt", "clothing"),
    ("summer dress", "clothing"),
    ("running shoes", "clothing"),
    ("leather sneakers", "clothing"),
    ("winter jacket", "clothing"),
    ("hoodie pullover", "clothing"),
    ("socks pack 6", "clothing"),
    ("underwear set", "clothing"),
    ("belt leather", "clothing"),
    ("handbag leather", "clothing"),
    ("wallet bifold", "clothing"),
    ("cap baseball", "clothing"),
    ("scarf wool", "clothing"),
    ("zara shirt", "clothing"),
    ("h&m jeans", "clothing"),

    # Electronics
    ("usb cable type c", "electronics"),
    ("phone charger 20w", "electronics"),
    ("earphone wired", "electronics"),
    ("bluetooth speaker", "electronics"),
    ("laptop bag", "electronics"),
    ("mouse wireless", "electronics"),
    ("keyboard mechanical", "electronics"),
    ("hdmi cable", "electronics"),
    ("memory card 64gb", "electronics"),
    ("power bank 10000mah", "electronics"),
    ("screen protector", "electronics"),
    ("phone case", "electronics"),
    ("adapter usb hub", "electronics"),
    ("webcam hd", "electronics"),

    # Utilities
    ("electricity bill", "utilities"),
    ("water bill payment", "utilities"),
    ("gas bill", "utilities"),
    ("internet broadband", "utilities"),
    ("wifi monthly", "utilities"),
    ("mobile recharge", "utilities"),
    ("airtel postpaid", "utilities"),
    ("jio plan", "utilities"),
    ("vodafone bill", "utilities"),
    ("society maintenance", "utilities"),

    # Education
    ("textbook mathematics", "education"),
    ("notebook 3 pack", "education"),
    ("pen ballpoint 10", "education"),
    ("pencil box", "education"),
    ("tuition fee", "education"),
    ("online course udemy", "education"),
    ("coursera subscription", "education"),
    ("school fee", "education"),
    ("exam registration", "education"),
    ("coaching class", "education"),
    ("workshop ticket", "education"),
    ("seminar registration", "education"),

    # Travel
    ("flight ticket economy", "travel"),
    ("airline boarding", "travel"),
    ("travel insurance", "travel"),
    ("visa fee", "travel"),
    ("passport renewal", "travel"),
    ("makemytrip booking", "travel"),
    ("cleartrip flight", "travel"),
    ("luggage check-in", "travel"),

    # Accommodation
    ("hotel room night", "accommodation"),
    ("resort stay 2 nights", "accommodation"),
    ("airbnb booking", "accommodation"),
    ("hostel dormitory", "accommodation"),
    ("room service", "accommodation"),
    ("hotel breakfast", "accommodation"),
    ("check-in fee", "accommodation"),

    # Personal Care
    ("shampoo head shoulders", "personal_care"),
    ("conditioner pantene", "personal_care"),
    ("face wash clean clear", "personal_care"),
    ("moisturizer spf30", "personal_care"),
    ("deodorant roll-on", "personal_care"),
    ("toothbrush electric", "personal_care"),
    ("toothpaste colgate", "personal_care"),
    ("razor gillette", "personal_care"),
    ("perfume eau de toilette", "personal_care"),
    ("body lotion", "personal_care"),
    ("sunscreen spf50", "personal_care"),
    ("makeup foundation", "personal_care"),
    ("lipstick matte", "personal_care"),
    ("salon haircut", "personal_care"),
    ("spa massage", "personal_care"),
    ("cotton pads", "personal_care"),
    ("tissue box", "personal_care"),

    # Stationery
    ("pen highlighter", "stationery"),
    ("stapler office", "stationery"),
    ("sellotape roll", "stationery"),
    ("a4 paper ream", "stationery"),
    ("folder plastic", "stationery"),
    ("whiteboard marker", "stationery"),
    ("ruler 30cm", "stationery"),
    ("eraser white", "stationery"),
    ("glue stick", "stationery"),
    ("scissors", "stationery"),

    # Others
    ("miscellaneous item", "others"),
    ("unknown charge", "others"),
    ("service fee", "others"),
    ("processing fee", "others"),
    ("convenience fee", "others"),

    # ── Extended samples (boost per-class coverage) ───────────────────────────
    ("grande latte starbucks", "food_dining"),
    ("zinger burger kfc", "food_dining"),
    ("cold brew coffee", "food_dining"),
    ("chicken tikka masala", "food_dining"),
    ("cheeseburger fries", "food_dining"),
    ("vanilla milkshake", "food_dining"),
    ("biryani chicken", "food_dining"),
    ("masala dosa plate", "food_dining"),
    ("mcmuffin egg", "food_dining"),
    ("fish and chips", "food_dining"),

    ("fresh vegetables 1kg", "groceries"),
    ("cooking oil 1l", "groceries"),
    ("fresh coriander bunch", "groceries"),
    ("paneer 500g", "groceries"),
    ("amul butter 500g", "groceries"),
    ("tata salt 1kg", "groceries"),
    ("britannia bread loaf", "groceries"),
    ("milk packet 1l", "groceries"),
    ("cabbage head", "groceries"),
    ("wheat atta 5kg", "groceries"),

    ("ola cab booking", "transport"),
    ("metro rail ticket", "transport"),
    ("rapido bike ride", "transport"),
    ("auto rickshaw charge", "transport"),
    ("city bus pass", "transport"),
    ("train local ticket", "transport"),
    ("ferry ride fee", "transport"),
    ("cab hire charge", "transport"),

    ("petrol fill up", "fuel"),
    ("diesel 20 liters", "fuel"),
    ("cng gas refill", "fuel"),
    ("hp petrol pump", "fuel"),
    ("iocl diesel fill", "fuel"),

    ("general physician visit", "healthcare"),
    ("orthopaedic consultation", "healthcare"),
    ("eye test ophthalmic", "healthcare"),
    ("dental scaling cleaning", "healthcare"),
    ("physiotherapy session knee", "healthcare"),
    ("ecg test hospital", "healthcare"),
    ("ultrasound scan fee", "healthcare"),

    ("cough syrup 100ml", "pharmacy"),
    ("vitamin d3 tablet strip", "pharmacy"),
    ("omega 3 capsule pack", "pharmacy"),
    ("antiseptic solution bottle", "pharmacy"),
    ("medplus prescription", "pharmacy"),
    ("cetrizine allergy tablet", "pharmacy"),

    ("pvr cinema ticket", "entertainment"),
    ("inox movie booking", "entertainment"),
    ("spotify monthly", "entertainment"),
    ("youtube premium plan", "entertainment"),
    ("ps4 controller gaming", "entertainment"),
    ("escape room ticket", "entertainment"),

    ("formal trousers black", "clothing"),
    ("polo tshirt", "clothing"),
    ("leather wallet slim", "clothing"),
    ("sports shoes running", "clothing"),
    ("winter coat jacket", "clothing"),
    ("bra innerwear set", "clothing"),
    ("kurta ethnic wear", "clothing"),

    ("mobile phone case cover", "electronics"),
    ("type c fast charger", "electronics"),
    ("wireless earbuds bluetooth", "electronics"),
    ("laptop cooling pad stand", "electronics"),
    ("hdmi 4k cable 2m", "electronics"),
    ("power bank 20000mah", "electronics"),

    ("broadband plan 100mbps", "utilities"),
    ("electricity bill payment", "utilities"),
    ("gas cylinder booking", "utilities"),
    ("airtel postpaid bill", "utilities"),
    ("jio fiber monthly plan", "utilities"),

    ("cbse textbook grade 10", "education"),
    ("entrance exam registration", "education"),
    ("online certification course", "education"),
    ("drawing notebook sketch", "education"),
    ("geometry set compass", "education"),

    ("indigo airlines flight", "travel"),
    ("spicejet ticket booking", "travel"),
    ("travel insurance premium", "travel"),
    ("yatra hotel booking", "travel"),
    ("visa fee application", "travel"),

    ("oyo room night stay", "accommodation"),
    ("hotel deluxe double room", "accommodation"),
    ("resort spa weekend", "accommodation"),
    ("hostel dorm bed", "accommodation"),

    ("dove shampoo 400ml", "personal_care"),
    ("gillette fusion razor", "personal_care"),
    ("lakme lipstick red", "personal_care"),
    ("nykaa foundation makeup", "personal_care"),
    ("hair colour black", "personal_care"),
    ("body wash shower gel", "personal_care"),

    ("stapler refill pins", "stationery"),
    ("whiteboard marker set", "stationery"),
    ("printing paper a4 ream", "stationery"),
    ("correction fluid pen", "stationery"),
    ("sticky notes pad", "stationery"),
]


def train_model(output_dir: str = "./models"):
    """Train and save the categorization model."""
    print("=" * 60)
    print("Smart Expense Tracker - ML Categorizer Training")
    print("=" * 60)

    texts, labels = zip(*TRAINING_DATA)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\n📊 Training data: {len(texts)} samples")
    print(f"📦 Categories: {len(le.classes_)}")
    print(f"   {', '.join(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.15, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),        
        min_df=1,
        max_features=8000,
        analyzer="word",
        sublinear_tf=True,         
    )

    clf = LogisticRegression(
        C=10.0,
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    clf.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    accuracy = (y_pred == y_test).mean()

    print(f"\n✅ Test Accuracy: {accuracy:.1%}")

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    cv_scores = cross_val_score(
        Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, max_features=8000)),
                  ("clf", LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs"))]),
        texts, y, cv=5, scoring="accuracy"
    )
    print(f"📈 CV Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    print("\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "categorizer.pkl")

    bundle = {
        "model": clf,
        "vectorizer": vectorizer,
        "label_encoder": le,
        "accuracy": float(accuracy),
        "cv_mean": float(cv_scores.mean()),
        "categories": list(le.classes_),
        "training_samples": len(texts),
    }
    joblib.dump(bundle, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    samples = [
        "Grande Latte Starbucks",
        "Petrol Shell 20L",
        "Doctor Consultation Fee",
        "Chicken Breast 500g",
        "Uber Ride Airport",
    ]
    print("\n🔍 Sample Predictions:")
    for sample in samples:
        vec = vectorizer.transform([sample.lower()])
        proba = clf.predict_proba(vec)[0]
        pred_idx = proba.argmax()
        predicted = le.inverse_transform([pred_idx])[0]
        confidence = proba[pred_idx]
        print(f"  '{sample}' → {predicted} ({confidence:.0%})")

    return model_path


if __name__ == "__main__":
    train_model()
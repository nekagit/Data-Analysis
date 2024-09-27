from flask import Flask, jsonify, request, abort

app = Flask(__name__)

# In-memory database
items_db = [
    {
        "id": 1,
        "name": "Item 1",
        "description": "nice thing",
        "price": 10.0,
        "available": True
    },
    {
        "id": 2,
        "name": "Item 2",
        "description": "cool thing",
        "price": 5.0,
        "available": False
    }
]

@app.route('/items', methods=['GET'])
def get_items():
    return jsonify(items_db)


if __name__ == '__main__':
    app.run(debug=True)


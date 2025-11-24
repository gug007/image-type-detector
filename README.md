source venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
python scripts/convert_tfjs.py

# To run the server
npm run dev

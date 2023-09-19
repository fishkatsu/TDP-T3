from bs4 import BeautifulSoup
import requests

url = 'https://www.swinburneonline.edu.au/faqs/'
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all the FAQ cards
    faq_cards = soup.find_all('div', class_='card')

    for index, card in enumerate(faq_cards, start=1):
        # Extract text content within the card
        card_text = card.get_text(separator=' ')
        
        # Split the text into question and answer
        parts = card_text.split('A.', 1)  # Split at 'A.' to separate question and answer
        if len(parts) == 2:
            question = parts[0].strip().replace('Q.', '').strip()
            answer = parts[1].strip()
            
            # Print the question and answer for each FAQ
            print(f'FAQ {index}:')
            print(f'Question: {question}')
            print(f'Answer: {answer}')
            print('-' * 50)

else:
    print('Failed to retrieve the web page. Status code:', response.status_code)

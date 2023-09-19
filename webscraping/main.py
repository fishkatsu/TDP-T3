from bs4 import BeautifulSoup
import requests
import csv

url = 'https://www.swinburneonline.edu.au/faqs/'
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all the FAQ cards
    faq_cards = soup.find_all('div', class_='card')
    
    # Initialize an empty list to store the extracted data
    faq_data = []

    for index, card in enumerate(faq_cards, start=1):
        # Extract text content within the card
        card_text = card.get_text(separator=' ')
        
        # Split the text into question and answer
        parts = card_text.split('A.', 1)  # Split at 'A.' to separate question and answer
        if len(parts) == 2:
            question = parts[0].strip().replace('Q.', '').strip()
            answer = parts[1].strip()

            # Append the extracted data to the list
            faq_data.append({"Question": question, "Answer": answer})

            # Print the question and answer for each FAQ
            print(f'FAQ {index}:')
            print(f'Question: {question}')
            print(f'Answer: {answer}')
            print('-' * 50)

    # Define the CSV file path
    csv_file_path = 'faq_data.csv'

    # Save the data to a CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["Question", "Answer"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(faq_data)

else:
    print('Failed to retrieve the web page. Status code:', response.status_code)

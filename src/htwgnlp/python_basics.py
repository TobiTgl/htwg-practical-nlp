"""Module for Python basics exercises.

This module contains some exercises to learn Python basics.
It covers some programming concepts and language features that will be useful for the course.
"""

import json
from collections import Counter


def get_even_numbers(numbers: list[int]) -> list[int]:
    """Returns a new list that contains only the even numbers.

    Use a list comprehension to solve this exercise.

    Args:
        numbers (list[int]): a list of numbers

    Returns:
        list[int]: a new list that contains only the even numbers

    Example:
        >>> get_even_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        [2, 4, 6, 8, 10]
    """
    return [number for number in numbers if number % 2 == 0]


def get_long_words(words: list[str]) -> list[str]:
    """Returns a new list that contains only the words that have more than 5 characters.

    Use a list comprehension to solve this exercise.

    Args:
        words (list[str]): a list of words

    Returns:
        list[str]: a new list that contains only the words that have more than 5 characters

    Example:
        >>> get_long_words(["apple", "banana", "cherry", "elderberry", "mango", "fig"])
        ["banana", "cherry", "elderberry"]
    """
    return [word for word in words if len(word) > 5]


def get_uppercase_words(words: list[str]) -> list[str]:
    """Returns a new list that contains the words in uppercase.

    Use a list comprehension to solve this exercise.

    Args:
        words (list[str]): a list of words

    Returns:
        list[str]: a new list that contains the words in uppercase

    Example:
        >>> get_uppercase_words(["apple", "banana", "cherry", "dates", "elderberry"])
        ["APPLE", "BANANA", "CHERRY", "DATES", "ELDERBERRY"]
    """
    return [word.upper() for word in words]


def build_phrases(adjectives: list[str], animals: list[str]) -> list[str]:
    """Returns a list of phrases by combining each adjective with each animal.

    This function takes two lists: one containing adjectives and the other containing animals.
    It returns a new list containing all possible combinations of adjectives and animals in the format "adjective animal".

    You should use a nested list comprehension to solve this exercise.

    Remember that you should not include empty strings in the output list.

    Args:
        adjectives (list of str): A list of adjectives.
        animals (list of str): A list of animals.

    Returns:
        list of str: A list containing all possible combinations of adjectives and animals.

    Example:
        >>> build_phrases(["big", "small", "furry", ""], ["cat", "dog", "rabbit", ""])
        ['big cat', 'big dog', 'big rabbit', 'small cat', 'small dog', 'small rabbit', 'furry cat', 'furry dog', 'furry rabbit']
    """
    return [
        f"{adjective} {animal}"
        for adjective in adjectives
        for animal in animals
        if adjective and animal
    ]


def get_word_lengths(words: list[str]) -> dict[str, int]:
    """Returns a dictionary with words as keys and their lengths as values.

    Use a dictionary comprehension to solve this exercise.

    Args:
        words (list of str): A list of words.

    Returns:
        dict: A dictionary where the keys are the words from the input list and the values are the lengths of those words.

    Example:
        >>> get_word_lengths(["apple", "banana", "cherry", "dates", "elderberry"])
        {'apple': 5, 'banana': 6, 'cherry': 6, 'dates': 5, 'elderberry': 11}
    """
    return {word: len(word) for word in words}


def print_product_price(product: str, price: int | float) -> str:
    """Returns a string that states the price of a given product.

    Note that the price should be formatted with two decimal places.

    Use f-string formatting to solve this exercise.

    Args:
        product (str): The name of the product.´
        price (int or float): The price of the product. Must be a positive number.

    Returns:
        str: A formatted string stating the price of the product in USD.

    Raises:
        ValueError: If the price is not a positive number.

    Example:
        >>> print_product_price("banana", 1.5)
        'The price of the product "banana" is 1.50 USD.'
    """
    if price <= 0:
        raise ValueError("Price must be a positive number.")
    return f"The price of {product} is {price:.2f} USD."


def count_purchases(purchases: list[str]) -> Counter:
    """Count the number of times each product was purchased.

    Args:
        purchases (list): A list of strings where each string represents a product purchased by a customer.

    Returns:
        Counter: A Counter object where the keys are the products and the values are the counts of each product.

    Example:
        >>> purchases = ["apple", "banana", "apple", "orange", "banana", "apple"]
        >>> count_purchases(purchases)
        Counter({'apple': 3, 'banana': 2, 'orange': 1})
    """
    return Counter(purchases)


def get_top_x_products(purchases: list[str], x: int) -> list[tuple[str, int]]:
    """Get the top 3 most popular products from a list of purchases.

    Args:
        purchases (list): A list of strings where each string represents a product purchased by a customer.
        x (int): The number of most popular products to return.

    Returns:
        list: A list of tuples where each tuple contains a product and its count,
              sorted by the most popular products in descending order.
              The list contains x tuples.

    Example:
        purchases = [
            "apple", "banana", "apple", "orange", "banana", "apple",
            "mandarin", "banana", "apple", "orange", "banana", "fig",
            "apple", "orange", "banana", "fig", "apple", "orange"
        ]
        get_top_x_products(purchases, 3)
        # Output: [('apple', 6), ('banana', 5), ('orange', 4)]
    """
    return Counter(purchases).most_common(x)


def sort_people_by_age(people: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Sort a list of people by their age.

    If two people have the same age, they should be sorted by their name in ascending order.

    Args:
        people (list of tuple): A list of tuples where each tuple contains a person's name (str) and age (int).

    Returns:
        list of tuple: The list of people sorted by age in ascending order.

    Example:
        >>> people = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
        >>> sort_people_by_age(people)
        [("Bob", 25), ("Alice", 30), ("Charlie", 35)]
    """
    return sorted(people, key=lambda person: (person[1], person[0]))


def write_dict_to_json_file(data: dict, filename: str) -> None:
    """Write the contents of a dictionary to a file in JSON format.

    Args:
        data (dict): The dictionary to write to the file.
        filename (str): The path to the file where the JSON data will be written.

    Example:
        data = {
            "name": "Alice",
            "age": 30,
            "city": "New York"
        }
        write_dict_to_json_file(data, 'output.json')
    """
    with open(filename, "w") as file:
        json.dump(data, file)


def read_dict_from_json_file(filename: str) -> dict:
    """Reads the contents of a JSON file and returns it as a dictionary.

    Args:
        filename (str): The path to the JSON file to be read.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(filename, "r") as file:
        return json.load(file)

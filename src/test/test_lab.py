"""
This file will contain test cases for the automatic evaluation of your
solution in main/lab.py. You should not modify the code in this file. You should
also manually test your solution by running main/app.py.
"""
import unittest
from langchain_core.outputs import LLMResult

from src.main.lab import classify
from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup


class TestLLMResponse(unittest.TestCase):
    """
        This function is a sanity check for the Language Learning Model (LLM) connection.
        It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
        it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
        operational before running tests and should not be modified without understanding the
        implications.
        Raises:
            Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
        """

    def test_llm_sanity_check(self):
        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")

    """
    Your prompt should make the LLM correctly classify positive responses.
    """

    def test_classify_positive_1(self):
        result = classify("that was a great movie")
        self.assertIn("positive", result)
        self.assertNotIn("negative", result)

    """
    Your prompt should make the LLM correctly classify negative responses.
    """

    def test_classify_negative_1(self):
        result = classify("that was a terrible movie")
        self.assertIn("negative", result)
        self.assertNotIn("positive", result)

    """
    Your prompt should make the LLM correctly classify positive responses.
    """

    def test_classify_positive_2(self):
        result = classify("i love that this product has so many interesting features")
        self.assertIn("positive", result)
        self.assertNotIn("negative", result)

    """
    Your prompt should make the LLM correctly classify negative responses.
    """

    def test_classify_negative_2(self):
        result = classify("i don't like this product")
        self.assertIn("negative", result)
        self.assertNotIn("positive", result)

    """
    Your prompt should make the LLM correctly classify positive responses.
    """

    def test_classify_positive_3(self):
        result = classify("i thought i wouldn't like the show, but i was pleasantly surprised")
        self.assertIn("positive", result)
        self.assertNotIn("negative", result)

    """
    Your prompt should make the LLM correctly classify negative responses.
    """

    def test_classify_negative_3(self):
        result = classify("the show wasn't very impressive")
        self.assertIn("negative", result)
        self.assertNotIn("positive", result)

if __name__ == '__main__':
    unittest.main()

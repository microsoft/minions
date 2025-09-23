import logging
import sys

# Configure logging to write to a file named 'calculator.log'.
# The 'exception' level is used to capture the full traceback.
logging.basicConfig(
    filename="calculator.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def divide(a, b):
    """
    A calculator function that performs division.
    It is intentionally designed to cause a ZeroDivisionError
    when the divisor `b` is zero.
    """
    try:
        logging.info(f"Attempting to divide {a} by {b}.")
        result = a / b
        logging.info(f"The result is: {result}")
        return result
    except ZeroDivisionError as e:
        # This block catches the specific ZeroDivisionError.
        print(
            "An error occurred during division. Please check calculator.log for details."
        )
        # The logging.exception() function logs the error and the full traceback.
        logging.exception("A deliberate ZeroDivisionError was encountered.")
        return None
    except Exception as e:
        # This is a general catch-all for any other potential errors.
        logging.error(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    # Example usage: This will work correctly.
    print("--- First Calculation (will succeed) ---")
    divide(10, 2)
    print("\n")

    # This call is intentionally designed to cause a ZeroDivisionError,
    # and the error will be logged to 'calculator.log'.
    print("--- Second Calculation (will fail on purpose) ---")
    divide(5, 0)
    print("\n")

    print("Program finished. Check 'calculator.log' for the error details.")

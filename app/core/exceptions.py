class ExpenseTrackerError(Exception):
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)
 
 
class OCRError(ExpenseTrackerError):
    def __init__(self, message: str):
        super().__init__(message, "OCR_ERROR")
 
 
class ParsingError(ExpenseTrackerError):
    def __init__(self, message: str):
        super().__init__(message, "PARSING_ERROR")
 
 
class CategorizationError(ExpenseTrackerError):
    def __init__(self, message: str):
        super().__init__(message, "CATEGORIZATION_ERROR")
 
 
class FileValidationError(ExpenseTrackerError):
    def __init__(self, message: str):
        super().__init__(message, "FILE_VALIDATION_ERROR")
 
 
class DatabaseError(ExpenseTrackerError):
    def __init__(self, message: str):
        super().__init__(message, "DATABASE_ERROR")
 
 
class StorageError(ExpenseTrackerError):
    def __init__(self, message: str):
        super().__init__(message, "STORAGE_ERROR")
package com.effective.detector.common.error

import com.effective.detector.common.util.logError
import com.effective.detector.common.util.logWarn
import jakarta.validation.ConstraintViolationException
import org.springframework.http.HttpStatus
import org.springframework.http.converter.HttpMessageNotReadableException
import org.springframework.security.access.AccessDeniedException
import org.springframework.web.ErrorResponse
import org.springframework.web.HttpRequestMethodNotSupportedException
import org.springframework.web.bind.MethodArgumentNotValidException
import org.springframework.web.bind.annotation.ExceptionHandler
import org.springframework.web.bind.annotation.RestControllerAdvice

@RestControllerAdvice
class ExceptionAdvice {
    @ExceptionHandler(HttpMessageNotReadableException::class)
    fun httpMessageNotReadableExceptionHandler(exception: HttpMessageNotReadableException): ErrorResponse {
        "{} : {}".logWarn(exception.message, exception)
        return ErrorResponse
            .builder(exception, HttpStatus.BAD_REQUEST, exception.message!!)
            .title(exception.javaClass.simpleName)
            .build()
    }

    @ExceptionHandler(HttpRequestMethodNotSupportedException::class)
    fun httpRequestMethodNotSupportedExceptionHandler(
        exception: HttpRequestMethodNotSupportedException,
    ): ErrorResponse {
        "{} : {}".logWarn(exception.message, exception)
        return ErrorResponse
            .builder(exception, HttpStatus.METHOD_NOT_ALLOWED, exception.message!!)
            .title(exception.javaClass.simpleName)
            .build()
    }

    @ExceptionHandler(MethodArgumentNotValidException::class)
    fun methodArgumentNotValidExceptionHandler(exception: MethodArgumentNotValidException): ErrorResponse {
        val objectErrors = exception.bindingResult.allErrors
        val errors: MutableList<String?> = ArrayList()
        for (objectError in objectErrors) {
            errors.add(objectError.defaultMessage)
        }
        val error = java.lang.String.join("\n", errors)
        "{} : {}".logWarn(exception.message, exception)
        return ErrorResponse
            .builder(exception, HttpStatus.BAD_REQUEST, error)
            .title(exception.javaClass.simpleName)
            .build()
    }

    @ExceptionHandler(ConstraintViolationException::class)
    fun constraintViolationExceptionHandler(exception: ConstraintViolationException): ErrorResponse {
        "{} : {}".logWarn(exception.message, exception)
        return ErrorResponse
            .builder(exception, HttpStatus.BAD_REQUEST, exception.message!!)
            .title(exception.javaClass.simpleName)
            .build()
    }

    @ExceptionHandler(AccessDeniedException::class)
    fun accessDeniedExceptionHandler(exception: AccessDeniedException): ErrorResponse {
        "{} : {}".logWarn(exception.message, exception)
        return ErrorResponse
            .builder(exception, HttpStatus.UNAUTHORIZED, exception.message!!)
            .title(exception.javaClass.simpleName)
            .build()
    }

    @ExceptionHandler(BusinessException::class)
    fun businessExceptionHandler(exception: BusinessException): ErrorResponse {
        val businessError = exception.error
        "{} : {}".logWarn(businessError.message, exception)
        return ErrorResponse
            .builder(exception, businessError.httpStatus, businessError.message)
            .title(businessError.name)
            .build()
    }

    @ExceptionHandler
    fun generalExceptionHandler(exception: Exception): ErrorResponse {
        "{} : {}".logError(exception.message, exception)
        return ErrorResponse
            .builder(exception, HttpStatus.INTERNAL_SERVER_ERROR, exception.message!!)
            .title(exception.javaClass.simpleName)
            .build()
    }
}

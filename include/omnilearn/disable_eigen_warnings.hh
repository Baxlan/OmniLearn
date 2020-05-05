// disable_eigen_warnings.hh

#ifndef OMNILEARN_DISABLE_EIGEN_WARNINGS_HH_
#define OMNILEARN_DISABLE_EIGEN_WARNINGS_HH_

#if defined(_MSC_VER) // Visual Studio
    #define DISABLE_WARNING_PUSH           __pragma(warning(push))
    #define DISABLE_WARNING_POP            __pragma(warning(pop))
    #define DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))

    #define DISABLE_WARNING_ALL             DISABLE_WARNING(26451)
    #define DISABLE_WARNING_EXTRA           DISABLE_WARNING(26812)
    #define DISABLE_WARNING_DEPRECATED_COPY DISABLE_WARNING(26812)
    #define DISABLE_WARNING_OLD_STYLE_CAST  DISABLE_WARNING(26450)
    #define DISABLE_WARNING_CONVERSION      DISABLE_WARNING(26495)

#elif defined(__GNUC__) || defined(__clang__) // GCC and CLANG
    #define DO_PRAGMA(X) _Pragma(#X)
    #define DISABLE_WARNING_PUSH           DO_PRAGMA(GCC diagnostic push)
    #define DISABLE_WARNING_POP            DO_PRAGMA(GCC diagnostic pop)
    #define DISABLE_WARNING(warningName)   DO_PRAGMA(GCC diagnostic ignored #warningName)

    #define DISABLE_WARNING_ALL              DISABLE_WARNING(-Wall)
    #define DISABLE_WARNING_EXTRA            DISABLE_WARNING(-W)
    #define DISABLE_WARNING_DEPRECATED_COPY  DISABLE_WARNING(-Wdeprecated-copy)
    #define DISABLE_WARNING_OLD_STYLE_CAST   DISABLE_WARNING(-Wold-style-cast)
    #define DISABLE_WARNING_CONVERSION       DISABLE_WARNING(-Wconversion)

#else
    #define DISABLE_WARNING_PUSH
    #define DISABLE_WARNING_POP

    #define DISABLE_WARNING_ALL
    #define DISABLE_WARNING_EXTRA
    #define DISABLE_WARNING_DEPRECATED_COPY
    #define DISABLE_WARNING_OLD_STYLE_CAST
    #define DISABLE_WARNING_CONVERSION

#endif

#endif // OMNILEARN_DISABLE_EIGEN_WARNINGS_HH_
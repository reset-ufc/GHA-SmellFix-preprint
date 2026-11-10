__app_name__ = "CI Smell Detector"
__version__ = "0.0.1"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    YAML_ERROR,
) = range(4)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    YAML_ERROR: "yaml parsing error"
}

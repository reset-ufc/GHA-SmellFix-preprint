from gha_ci_detector import __app_name__, cli


def main():
    print("Welcome to GHA CI Detector")
    cli.app(prog_name=__app_name__)


if __name__ == "__main__":
    main()

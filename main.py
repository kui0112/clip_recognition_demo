import click

from config import Config
from main_context import MainContext
from summarize_processor import SummarizeProcessor
from test_ui import setup_test_ui


@click.command()
@click.option('--config', default='config.json', help='Path to the configuration file (default: config.json).')
@click.option('--test', is_flag=True, default=False, help='Run in test mode.')
def main(config: str, test: bool):
    c = Config().parse(config)
    print("parse config file success.")
    if test:
        print("setup test ui.")
        setup_test_ui(c)
    else:
        print("setup context.")
        ctx = MainContext(c)
        ctx.set_processors([SummarizeProcessor(c, network_enabled=c.enable_network_notify)])
        ctx.run()


if __name__ == '__main__':
    main()

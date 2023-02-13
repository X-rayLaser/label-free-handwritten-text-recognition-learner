import os
import shutil

from .base import Command


class FontsCommand(Command):
    name = 'fonts'
    help = 'Extracts pseudo-handwritten fonts from Google fonts folder'

    def configure_parser(self, parser):
        configure_parser(parser)

    def __call__(self, args):
        run(args)


def configure_parser(parser):
    parser.add_argument('fonts_dir', type=str,
                        help='Path to the location of google fonts directory')
    parser.add_argument('output_dir', type=str,
                        help='Path to the output directory')
    parser.add_argument('--num_fonts', type=int,
                        default=100, help='Maximum number of fonts to take')
    parser.add_argument(
        '--lang', type=str, default='latin',
        help='Specifies target language that fonts are required to support'
    )


def run(args):
    ofl_dir = os.path.join(args.fonts_dir, 'ofl')
    apache_dir = os.path.join(args.fonts_dir, 'apache')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    found_fonts = copy_fonts(ofl_dir, args.lang, args.output_dir,
                             max_fonts=args.num_fonts, found_fonts=0)
    copy_fonts(apache_dir, args.lang, args.output_dir,
               max_fonts=args.num_fonts, found_fonts=found_fonts)


def copy_fonts(src_dir, lang, output_dir, max_fonts, found_fonts=0):
    for font_dir in os.listdir(src_dir):
        if found_fonts >= max_fonts:
            break

        dir_path = os.path.join(src_dir, font_dir)
        ttf_files = [f for f in os.listdir(dir_path) if f.endswith('.ttf')]
        if not ttf_files:
            print(f'No ttf files found', os.listdir(dir_path))
            continue

        metadata_path = os.path.join(dir_path, 'METADATA.pb')
        if not os.path.exists(metadata_path):
            print('no metadata file, directory', font_dir, 'dir path',
                  dir_path, 'dir content', os.listdir(dir_path))
            continue

        with open(metadata_path) as f:
            s = f.read()

        if 'category: "HANDWRITING"' in s and f'subsets: "{lang}"' in s:
            print('found font', font_dir)
            found_fonts += 1

            for name in ttf_files:
                src = os.path.join(dir_path, name)
                dest = os.path.join(output_dir, name)
                shutil.copyfile(src, dest)

    return found_fonts

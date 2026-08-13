from psf_tools.library import build_psf_library


def main():
    target_dir = "star_psfs_separated"
    variants = [1, 2, 3, 4, 5, 6]
    dead_fields = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 24, 25, 30, 31, 37, 38, 44, 45, 46, 47, 48, 49]

    for variant in variants:
        try:
            build_psf_library(target_dir, variant, dead_fields=dead_fields)
        except Exception as e:
            print(f"Error building variant {variant}: {e}")
            continue


if __name__ == '__main__':
    main()


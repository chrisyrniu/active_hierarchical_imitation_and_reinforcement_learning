import argparse

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
    )

    parser.add_argument(
        '--expnum',
        required=True,
        type=int,
        help='Include to reset policy'
    )

    parser.add_argument(
        '--reset_maze',
        '-rm',
        action='store_true',
        help='Reset maze to HAC'
    )

    parser.add_argument(
        '--randomize',
        action='store_true',
        help='Include to randomize maze'
    )

    parser.add_argument(
        '--beta',
        '-b',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Include to fix current policy'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Include to visualize training'
    )

    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Include to use training mode only'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--all_trans',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--hind_action',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--penalty',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--prelim_HER',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--HER',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--Q_values',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--active_learning',
        type=int,
        default=0,
        help='Include to use active learning'
        # if active_learning == 0: no active learning
        # if active_learning == 1: noise_based active learning
        # if active_learning == 2: multi-policy active learning
    )  

    parser.add_argument(
        '--results',
        action='store_true',
        help='Include to generate test results'
    )       

    parser.add_argument(
        '--al_test',
        type=int,
        default=0,
        help='Include to generate active learning test results'
        # if al_test == 0: no active learning
        # if al_test == 1: noise_based active learning
        # if al_test == 2: multi-policy active learning
    )  

    FLAGS, unparsed = parser.parse_known_args()


    return FLAGS

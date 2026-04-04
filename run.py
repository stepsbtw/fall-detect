from src.pipeline import cross_sensor_experiment, fused_missing_experiment, parse_args, train_experiment


def main():
    args = parse_args()
    actions = [args.train, args.cross_sensor, args.fused_missing]
    if not any(actions):
        raise SystemExit('select at least one action: --train, --cross_sensor, or --fused_missing')
    if args.train:
        train_experiment(args)
    if args.cross_sensor:
        cross_sensor_experiment(args)
    if args.fused_missing:
        fused_missing_experiment(args)


if __name__ == '__main__':
    main()

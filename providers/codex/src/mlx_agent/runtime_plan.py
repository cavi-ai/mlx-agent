"""What a runtime adds to a serving plan: argv, recipe values, and plan fields.

Splash and Strata each return one contribution. Task endpoints return earlier
through task_serving.plan_start. Chat runtimes add nothing here.
"""


class Contribution:
    def __init__(self, argv, values, fields):
        self.argv = list(argv)
        self.values = dict(values)
        self.fields = dict(fields)

import os
import random

from locust import task, constant_throughput, LoadTestShape, TaskSet, HttpUser


PRODUCTS = [
    '0PUK6V6EV0', '1YMWWN1N4O', '2ZYFJ3GM2N', '66VCHSJNUP', '6E92ZMYYFZ',
    '9SIQT8TOJO', 'L9ECAV7KIM', 'LS4PSXUNUM', 'OLJCESPC7Z',
]


def index(l):
    l.client.get("/")


def set_currency(l):
    l.client.post("/setCurrency",
                  {'currency_code': random.choice(['EUR', 'USD', 'JPY', 'CAD'])})


def browse_product(l):
    l.client.get("/product/" + random.choice(PRODUCTS))


def view_cart(l):
    l.client.get("/cart")


def add_to_cart(l):
    product = random.choice(PRODUCTS)
    l.client.get("/product/" + product)
    l.client.post("/cart", {
        'product_id': product,
        'quantity': random.choice([1, 2, 3, 4, 5, 10]),
    })


def checkout(l):
    add_to_cart(l)
    with l.client.post("/cart/checkout", {
        'email': 'someone@example.com',
        'street_address': '1600 Amphitheatre Parkway',
        'zip_code': '94043',
        'city': 'Mountain View',
        'state': 'CA',
        'country': 'United States',
        'credit_card_number': '4432-8015-6152-0454',
        'credit_card_expiration_month': '1',
        'credit_card_expiration_year': '2039',
        'credit_card_cvv': '672',
    }, catch_response=True) as resp:
        if resp.status_code in (422, 500):
            resp.success()


class UserBehavior(TaskSet):
    def on_start(self):
        index(self)

    tasks = {
        index: 1,
        set_currency: 2,
        browse_product: 10,
        add_to_cart: 2,
        view_cart: 3,
        checkout: 4,
    }


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    # 3 req/s per user gives ~240 req/s at MID(80 users), which drives
    # recommendation to ~350 mc — close to the training mean of 368 mc.
    wait_time = constant_throughput(3)


class CyclicStepShape(LoadTestShape):
    """Deterministic low→mid→high→mid step cycle, repeated for the whole run.

    Time-based (locust run time), identical and reproducible for every algorithm.
    Returns a constant tuple within each step; never returns None so the test runs
    indefinitely until the eval stops the load generator.
    """

    min_users = int(os.environ.get("LOAD_MIN_USERS", "10"))
    max_users = int(os.environ.get("LOAD_MAX_USERS", "150"))
    step_seconds = int(os.environ.get("LOAD_STEP_SECONDS", "30"))
    spawn_rate = int(os.environ.get("LOAD_SPAWN_RATE", "20"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mid = (self.min_users + self.max_users) // 2
        self._levels = [self.min_users, mid, self.max_users, mid]
        self._cycle = self.step_seconds * len(self._levels)

    def tick(self):
        run_time = self.get_run_time()
        idx = int((run_time % self._cycle) // self.step_seconds)
        return (self._levels[idx], self.spawn_rate)

"""Async API calls to OpenAI GPTs.

Reference: https://github.com/SpellcraftAI/oaib

```python
# Pass API key, or set env var `OPENAI_API_KEY`
batch = Batch(tpm=10000, azure=AzureConfig())

# Add chat completion requests
for i in range(5):
    await batch.add(
        "chat.completions.create",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "say hello"}]
    )

# Run the batch processing for chat completions
chat = await batch.run()

# Add embedding requests
for i in range(5):
    await batch.add(
        "embeddings.create",
        model="text-embedding-3-large",
        input="hello world"
    )

# Run the batch processing for embeddings
embeddings = await batch.run()
```
"""
import os
import signal
from asyncio import (FIRST_COMPLETED, CancelledError, Event, Lock, Queue,
                     QueueEmpty, create_task, gather, sleep,
                     wait, wait_for)
from asyncio import TimeoutError as AsyncTimeoutError
from datetime import datetime
from os import environ
from time import time
from types import SimpleNamespace
from typing import Coroutine, List, Set, Dict

import openai
import pandas as pd
from tqdm.auto import tqdm


async def add_messages_and_run(batch, messages, model):
    """Help async function to add messages to the batch and run it."""
    for pid, msg in messages:
        await batch.add("chat.completions.create",
                        model=model,
                        messages=msg,
                        metadata={"pid": pid})
    return await batch.run()


def create_azure_config(config):
    return AzureConfig(
        azure_endpoint=config["azure_endpoint"],
        api_version=config["api_version"],
        api_key=config["api_key"]
    )


class AzureConfig(SimpleNamespace):
    """Azure OpenAI API configuration."""

    def __init__(
            self,
            azure_endpoint: str = environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version: str = environ.get("AZURE_OPENAI_VERSION"),
            api_key: str = environ.get("AZURE_OPENAI_API_KEY"),
    ):
        """Initialize AzureConfig with the Azure OpenAI API credentials."""
        super().__init__(azure_endpoint=azure_endpoint,
                         api_version=api_version,
                         api_key=api_key)

    def display_endpoint(self):
        """Display the Azure endpoint."""
        print(f"Endpoint: {self.azure_endpoint}")

    def is_configured(self):
        """Check if all necessary API configuration details are provided."""
        return all([self.azure_endpoint, self.api_version, self.api_key])


def getattr_dot(obj: any, index: str):
    """Get an attribute of an object using dot notation."""
    parts = index.split('.')
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


async def cancel_all(tasks: set):
    """Cancel all tasks in a set."""
    tasks.discard(None)
    for task in tasks:
        task.cancel()

    try:
        await gather(*tasks)
    except CancelledError:
        pass


async def race(tasks: Set[Coroutine]):
    """Run tasks concurrently and return when the first one completes."""
    tasks = map(create_task, tasks)
    done, pending = await wait(tasks, return_when=FIRST_COMPLETED)

    await cancel_all(pending)
    results = await gather(*done)

    return results[0]


async def close_queue(queue: Queue):
    """Close the queue and wait for all tasks to complete."""
    while not queue.empty():
        queue.get_nowait()
        queue.task_done()

    return await queue.join()


def get_limits(headers):
    """Get the rate limits from the response headers."""
    rpm = headers.get("x-ratelimit-limit-requests")
    tpm = headers.get("x-ratelimit-limit-tokens")

    return int(rpm), int(tpm)


class Batch:
    """
    An instance for managing batch requests to the OpenAI API.

    Parameters
    ----------
    rpm : int, default: `500`
        The maximum number of requests allowed per minute. Defaults to lowest
        tier.
    tpm : int, default: `10_000`
        The maximum number of tokens allowed per minute. Defaults to lowest
        tier.
    workers : int, default: `8`
        The number of concurrent workers to process the requests.
    safety : float, default: `0.1`
        The safety factor to apply to the token per minute calculation.
        Defaults to `0.1`, which means the engine will wait
        until the current TPM drops below 90% of the limit,
        to prevent going over. This is necessary because
        we don't know how many tokens a response will contain before we get it.
    loglevel : int, default: `1`
        If set to 0, suppresses the progress bar and logging output.
        If set to 1, logs include metadata only.
        If set to 2, logs include both data and
        metadata for each request.
    timeout : int, default: `60`
        The maximum time to wait for a single request to complete, in seconds.
    api_key : str, default: `os.environ.get("OPENAI_API_KEY")`
        The API key used for authentication with the OpenAI API. If not
        provided, the class attempts to use an API_KEY constant defined
        elsewhere.
    log_path : str, default: `"oaib.txt"`
        The file path for logging the progress and errors of batch processing.
        Defaults to "oaib.txt".
    **client_args
        Additional keyword arguments to pass to the OpenAI client.
    """

    def __init__(self,
                 rpm: int = -1,
                 tpm: int = 10_000,
                 workers: int = 8,
                 safety: float = 0.1,
                 loglevel: int = 1,
                 timeout: int = 60,
                 azure=None,
                 api_key: str = None,
                 logdir: str = "oaib.txt",
                 index: List[str] = None,
                 **client_kwargs):
        """Init method for the Batch class."""
        # print(f"azure: {azure}")
        # api_key = api_key or (os.environ.get("AZURE_OPENAI_API_KEY")
        #                       if azure else os.environ.get("OPENAI_API_KEY"))

        api_key = api_key or (azure.api_key
                              if azure else os.environ.get("OPENAI_API_KEY"))


        if not api_key:
            raise ValueError(
                "No OpenAI API key found.\
                Please provide an `api_key` parameter or\
                set the `OPENAI_API_KEY` environment variable."
            )

        if loglevel > 2:
            raise ValueError(
                f"Allowable `loglevel` values are 0, 1, or 2; found {loglevel}"
            )

        self.rpm = rpm
        self.tpm = tpm
        self.safety = safety
        self.loglevel = loglevel
        self.timeout = timeout
        self.logdir = logdir
        self.index = index
        self.azure = None

        if rpm == -1:
            self.rpm = tpm // 1000 * 6

        if azure:
            azure = vars(azure)
            self.azure = azure

            self.client = openai.AsyncAzureOpenAI(
                **{
                    **azure, "api_key": api_key
                }, **client_kwargs)
        else:
            self.client = openai.AsyncOpenAI(api_key=api_key, **client_kwargs)

        self.__num_workers = workers

        self.__lock = Lock()
        self.__queue = Queue()
        self.__stopped = Event()

        self.__workers = set()
        self.__processing = set()
        self.__callbacks = set()

        self.__current = SimpleNamespace(rpm=0, tpm=0)
        self.__totals = SimpleNamespace(requests=0, tokens=0, queued=0)
        self.__progress = SimpleNamespace(main=None, rpm=None, tpm=None)

        # Send stop event on SIGINT.
        signal.signal(signal.SIGINT,
                      lambda code, stack: create_task(self.stop(code, stack)))

    def __clear_log(self):
        """Clear the log file."""
        with open(self.logdir, "w", encoding="utf-8") as file:
            file.write("")

    def log(self, *messages, worker: int = None, loglevel: int = None):
        """Log messages to the log file."""
        if (loglevel or self.loglevel) > 0:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            for message in messages:
                prefix = f"WORKER {worker}" if worker else "MAIN"
                message = " | ".join([prefix.rjust(8), timestamp, message])

                with open(self.logdir, "a", encoding="utf-8") as file:
                    file.write(message + "\n")

    async def _cleanup(self):
        """
        Ensure the stop event is set.

        All workers and processing tasks are
        cancelled. Also closes the progress bar and queue.
        """
        # Stop was triggered, tasks done: final tick.
        self._tick()

        if self.succeeded:
            self.log("WAITING FOR CLOCK")
            await wait([self.__clock])
        else:
            self.log("CANCELLING ALL TASKS")
            await cancel_all({
                self.__clock,
                *self.__processing,
                *self.__callbacks,
                *self.__workers,
            })

        await close_queue(self.__queue)

        self.__current = SimpleNamespace(rpm=0, tpm=0)
        self.__totals = SimpleNamespace(requests=0, tokens=0, queued=0)

        for bar_progress in vars(self.__progress).values():
            if isinstance(bar_progress, tqdm):
                bar_progress.close()

        self.__workers.clear()

    def _tick(self):
        now = time()
        seconds = now - self._start
        minutes = seconds / 60

        if self._last_tick and now - self._last_tick < pd.Timedelta("1s"):
            return

        self.__current.rpm = self.__totals.requests // minutes
        self.__current.tpm = self.__totals.tokens // minutes

        self.__progress.main.n = self.__totals.requests
        self.__progress.main.total = self.__totals.queued

        self.__progress.rpm.n = self.__current.rpm
        self.__progress.rpm.total = self.rpm

        self.__progress.tpm.n = self.__current.tpm
        self.__progress.tpm.total = self.tpm

        self.__progress.main.refresh()
        self.__progress.tpm.refresh()
        self.__progress.rpm.refresh()

        if self.__stopped.is_set():
            self.__progress.main.set_description(
                "✅ DONE" if self.succeeded else "🛑 STOPPED", refresh=True)

    async def _watch(self):
        while True:
            await sleep(0.1)
            self._tick()

            if self.__stopped.is_set():
                break

    async def _process(self, request, i=None):
        endpoint, func, kwargs, metadata = request

        if self.loglevel == 1:
            log_content = f"{metadata}"
        else:
            log_content = f"{metadata} | {kwargs}"

        self.log(f"PROCESSING | {log_content}", worker=i)

        try:
            [response] = await wait_for(gather(func(**kwargs)),
                                        timeout=self.timeout)

        except AsyncTimeoutError:
            self.log(f"TIMEOUT | {self.timeout}s | {kwargs}", worker=i)
            return

        except Exception as e:
            self.log(f"PROCESSING ERROR | {e}", worker=i)
            return

        headers = response.headers
        response = response.parse()
        tokens = response.usage.total_tokens

        row = pd.DataFrame([{
            **metadata, "endpoint": endpoint,
            **kwargs, "result": response.model_dump()
        }])

        # Store one copy of response headers - for use by Auto subclass.
        if self._headers is None:
            self.log(f"HEADERS | {dict(headers)}")
            self._headers = headers

        self.__totals.requests += 1
        self.__totals.tokens += tokens

        self.output = pd.concat([self.output, row], ignore_index=True)
        self.log(f"PROCESSED | {kwargs}", worker=i)

        if self._callback:
            callback = create_task(self._callback(row))
            self.__callbacks.add(callback)
            callback.add_done_callback(
                lambda _: self.__callbacks.remove(callback))

    def _next(self, i):
        try:
            self.log(f"REQUESTS: {self.__queue.qsize()}", worker=i)
            request = self.__queue.get_nowait()
            self.__queue.task_done()

        except QueueEmpty:
            self.log("EMPTY QUEUE", worker=i)

            if self._listening:
                self.log("LISTENING", worker=i)
                return True

            return False

        processing = create_task(self._process(request, i))
        self.__processing.add(processing)
        processing.add_done_callback(
            lambda _: self.__processing.remove(processing))

        return True

    async def __worker(self, i):
        while True:
            async with self.__lock:
                if self.__stopped.is_set():
                    break

                proceed = self._next(i)
                if not proceed:
                    break

                self.__progress.main.set_description("🟢 RUNNING",
                                                     refresh=True)

                now = time()
                avg_tpr = (now - self._start) / (self.__totals.requests or 1)

                # The RPM does not need a safety threshold because it is known
                # in advance, but we still apply a 1% reduction to minimize
                # going over on small timescales.
                # effective_rpm = 0.99 * self.rpm
                effective_tpm = (1 - self.safety) * self.tpm
                rpm_delay = 60 / self.rpm

                start = now
                while self.__current.tpm + avg_tpr >= effective_tpm \
                        and not self.__stopped.is_set():
                    self.__progress.main.set_description("🟡 WAITING",
                                                         refresh=True)

                    await sleep(0.1)
                end = time()

                remaining = rpm_delay - (end - start)
                if remaining > 0:
                    await sleep(remaining)

    def __setup(self, callback=None, listening=False):
        self.output = pd.DataFrame()
        self.succeeded = False

        # For use by Auto(Batch) subclass.
        self._headers = None
        # If true, clearing the queue will not stop the engine.
        self._listening = listening
        # Callback to run on each processed request.
        self._callback = callback

        self._start = time()
        self._last_tick = None

        self.__clear_log()
        self.__stopped.clear()
        self.__clock = create_task(self._watch())

        self.__workers = {
            create_task(self.__worker(i))
            for i in range(self.__num_workers)
        }

        silence = self.loglevel == 0

        self.__progress.main = tqdm(total=self.__queue.qsize(),
                                    unit='req',
                                    dynamic_ncols=True,
                                    disable=silence)

        self.__progress.rpm = tqdm(
            desc="RPM",
            total=self.rpm,
            unit='rpm',
            dynamic_ncols=True,
            disable=silence,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

        self.__progress.tpm = tqdm(
            desc="TPM",
            total=self.tpm,
            unit='tpm',
            dynamic_ncols=True,
            disable=silence,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

        if self.azure:
            self.log(f"USING AZURE | {self.azure}")

    async def listen(self, callback=None):
        """
        Listen for incoming requests and processes them as they arrive.

        This method is non-blocking and
        can be used to process requests in real-time.
        """
        self.__setup(callback, True)
        await self.__stopped.wait()
        await self.stop()

    async def run(self, callback=None):
        """
        Initiate the processing of all queued requests.

        Manage requests according to the specified rate limits.
        This method waits for the processing to complete or stop conditionally.

        Returns
        -------
        output : pandas.DataFrame
            A DataFrame containing the results of the processed requests.
        """
        if not self.__queue.qsize() >= 1:
            raise ValueError(
                "Engine cannot run without at least one job scheduled")

        # start = time()
        # Run setup and wait for a stop condition.
        self.__setup(callback)
        await race({self.__queue.join(), self.__stopped.wait()})

        # If the run was successful, it needs to be stopped. Finish processing
        # existing requests first.
        if not self.__stopped.is_set():
            self.log("FINISHING PROCESSING | 5 second timeout")
            await gather(*self.__processing)
            await gather(*self.__callbacks)
            await gather(*self.__workers)
            await self.stop()

        if self.index:
            self.log("INDEX | Setting index")
            self.output.set_index(self.index, inplace=True)
            self.output.sort_index(inplace=True)

        self.log("RETURNING OUTPUT")
        # print(f"\nRun took {time() - start:.2f}s.\n")
        return self.output

    async def stop(self, code=0, stack=None):
        """
        Stop the processing of requests.

        Parameters
        ----------
        code : int, default: `None`
            Set an exit code. By default,
            it is set to 1 if the run was interrupted before completing,
            0 otherwise.
        stack : never
            Reserved for future use,
            e.g., for passing exception stack information. Currently not used.

        Returns
        -------
        success : bool
            False if the run was cancelled or interrupted,
            True if it completed successfully.
        """
        self.succeeded = code == 0

        self.log(f"STOP EVENT | Exit code {code}")
        if stack:
            self.log(f"STACK INFO\n\n{stack}\n")

        self.__stopped.set()
        await self._cleanup()

    async def add(self,
                  endpoint="chat.completions.create",
                  metadata: dict = {},
                  **kwargs):
        """
        Schedules an API request to be added to the batch processing queue.

        Parameters
        ----------
        endpoint : str, default: `"chat.completions.create"`
            The OpenAI API endpoint to call,
            e.g., 'chat.completions.create' or 'embeddings.create'.
        metadata : dict, default: `None`
            A dictionary containing additional data to be added
            to this observation row in the DataFrame.
        **kwargs
            Keyword arguments to pass to the OpenAI API endpoint function.
            Common kwargs include 'model' and input parameters,
            like 'messages' for 'chat.completions.create'
            or 'input' for 'embeddings.create'.

        Returns
        -------
        None
        """
        # Read the client method.
        func = getattr_dot(self.client.with_raw_response, endpoint)

        # Add the request to the queue.
        request = (endpoint, func, kwargs, metadata)
        model = kwargs.get("model")
        await self.__queue.put(request)

        self.__totals.queued += 1
        self.log(f"QUEUED | {model}")

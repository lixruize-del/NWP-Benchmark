# GHCNh hourly data downloader (ported from StationCast `dataset/downloader.py`).
# https://data.ecmwf.int/forecasts/20221231/12z/0p4-beta/oper/20221231120000-102h-oper-fc.grib2
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="CDS API client parameters")
    parser.add_argument("--proxy", type=str, default="direct")
    parser.add_argument("--s3", type=bool, default=False)
    parser.add_argument("--hour", type=str, default="06z")
    parser.add_argument("--worker_num", type=int, default=500)
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.ncei.noaa.gov/data/global-historical-climatology-network/hourly/access/by-station/",
        required=False,
    )
    return parser.parse_args()


def bytes_to_string(n):
    u = ["", "K", "M", "G", "T", "P"]
    i = 0
    while n >= 1024:
        n /= 1024.0
        i += 1
    return "%g%s" % (int(n * 10 + 0.5) / 10.0, u[i])


def parserHtml(url, finishedContents):
    result = []
    try:
        response = urllib.request.urlopen(url)
        string = response.read()
        html = string.decode("utf-8")
        soup = BeautifulSoup(html, "html.parser")
        invalidLink1 = "#"
        invalidLink2 = "javascript:void(0)"
        for k in soup.find_all("a"):
            link = k.get("href")
            if link is None:
                continue
            if link == invalidLink1:
                pass
            elif link == invalidLink2:
                pass
            elif link.find("javascript:") != -1:
                pass
            elif link in finishedContents:
                pass
            else:
                result.append(link)
        result.sort()
        return result, True
    except Exception:
        return result, False


class HRES_Client(object):
    logger = logging.getLogger("cdsapi")

    def __init__(
        self,
        quiet=False,
        debug=False,
        verify=None,
        timeout=60,
        progress=True,
        full_stack=False,
        delete=True,
        retry_max=500,
        sleep_max=120,
        proxy="special",
        s3=False,
    ):
        if not quiet:
            level = logging.DEBUG if debug else logging.INFO
        else:
            level = logging.WARNING
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
        self.retry_max = retry_max
        self.timeout = timeout
        self.sleep_max = sleep_max
        self.verify = False
        self.progress = progress and not quiet
        self.session = requests.Session()
        self.proxy_seting(proxy)

    def proxy_seting(self, proxy):
        if proxy == "direct":
            print("you does not use any proxy !!!!")
        elif proxy == "special":
            # Do not hard-code credentials in the repo. Set e.g.:
            #   export NWP_HTTP_PROXY='http://user:pass@host:port/'
            #   export NWP_HTTPS_PROXY='http://user:pass@host:port/'  # optional
            http_p = os.environ.get("NWP_HTTP_PROXY")
            https_p = os.environ.get("NWP_HTTPS_PROXY", http_p)
            if not http_p:
                raise ValueError(
                    "proxy=='special' requires NWP_HTTP_PROXY (and optionally NWP_HTTPS_PROXY) "
                    "in the environment; do not commit proxy passwords to git."
                )
            proxies = dict(http=http_p, https=https_p or http_p)
            print("you use special proxy (from NWP_HTTP_PROXY) !!!!")
            self.session.proxies.update(proxies)
        else:
            raise ValueError("proxy type must be 'direct' or 'special' !!!! ")

    def content_info(self, url):
        r = requests.head(url)
        reply = dict(
            location=url,
            content_length=r.headers.get("Content-Length", 0),
            content_type=r.headers["Content-Type"],
        )
        return reply

    def local_downloader(self, url, target):
        if target is None:
            target = url.split("/")[-1]

        size = int(self.content_info(url).get("content_length"))

        self.info("Downloading %s to %s (%s)", url, target, bytes_to_string(size))
        start = time.time()

        mode = "wb"
        total = 0
        sleep = 10
        tries = 0
        headers = None
        complete = False
        if os.path.exists(target):
            total = os.path.getsize(target)

            if total == size:
                self.info(
                    "File %s had been downloaded to %s completely  %s/s",
                    url,
                    target,
                    bytes_to_string(size),
                )
                complete = True
            else:
                os.remove(target)
                total = 0
                self.warning("Resuming download at byte %s" % (total,))

        if not complete:
            if total > 0:
                self.error(
                    "Download incomplete, downloaded %s byte(s) out of %s" % (total, size)
                )
                self.warning("Sleeping %s seconds" % (sleep,))
                time.sleep(sleep)
                mode = "ab"
                total = os.path.getsize(target)
                sleep *= 1.5
                if sleep > self.sleep_max:
                    sleep = self.sleep_max
                headers = {"Range": "bytes=%d-" % total}
                tries += 1
                self.warning("Resuming download at byte %s" % (total,))

            while tries < self.retry_max:
                r = self.robust(self.session.get)(
                    url,
                    stream=True,
                    verify=self.verify,
                    headers=headers,
                    timeout=self.timeout,
                )
                try:
                    r.raise_for_status()

                    with tqdm(
                        total=size,
                        unit_scale=True,
                        unit_divisor=1024,
                        unit="B",
                        disable=not self.progress,
                        leave=False,
                    ) as pbar:
                        pbar.update(total)
                        with open(target, mode) as f:
                            for chunk in r.iter_content(chunk_size=1024):
                                if chunk:
                                    try:
                                        f.write(chunk)
                                        total += len(chunk)
                                        pbar.update(len(chunk))
                                    except Exception:
                                        pass
                except requests.exceptions.ConnectionError as e:
                    self.error("Download interupted: %s" % (e,))
                finally:
                    r.close()

                if total != size:
                    raise Exception(
                        "Download failed: downloaded %s byte(s) out of %s" % (total, size)
                    )
                self.info("Download sucessful: downloaded %s byte(s) out of %s" % (total, size))
                break
        elapsed = time.time() - start
        if elapsed:
            self.info("Download rate %s/s", bytes_to_string(size / elapsed))

    def untar_files(self, file_list):
        for target in file_list:
            print(f"Untar processing: {target}")
            if self.is_tarfile(target):
                tar = tarfile.open(target)
                dirname = os.path.dirname(target).replace("Tar", "Untar")
                os.makedirs(dirname, exist_ok=True)
                full_date = os.path.basename(target).split(".")[1]
                yy = full_date[:4]
                mm_dd = f"{full_date[4:6]}-{full_date[6:8]}"
                untar_path = f"{dirname}/{yy}-{mm_dd}"
                print(f"Untar  {target} to {untar_path}")
                tar.extractall(path=untar_path)
                time.sleep(1)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def robust(self, call):
        def retriable(code, reason):
            if code in [
                requests.codes.internal_server_error,
                requests.codes.bad_gateway,
                requests.codes.service_unavailable,
                requests.codes.gateway_timeout,
                requests.codes.too_many_requests,
                requests.codes.request_timeout,
            ]:
                return True
            return False

        def wrapped(*args, **kwargs):
            tries = 0
            while tries < self.retry_max:
                try:
                    r = call(*args, **kwargs)
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                ) as e:
                    r = None
                    self.warning(
                        "Recovering from connection error [%s], attemps %s of %s",
                        e,
                        tries,
                        self.retry_max,
                    )

                if r is not None:
                    if not retriable(r.status_code, r.reason):
                        return r
                    try:
                        self.warning(r.json()["reason"])
                    except Exception:
                        pass
                    self.warning(
                        "Recovering from HTTP error [%s %s], attemps %s of %s",
                        r.status_code,
                        r.reason,
                        tries,
                        self.retry_max,
                    )

                tries += 1
                self.warning("Retrying in %s seconds", self.sleep_max)
                time.sleep(self.sleep_max)
                self.info("Retrying now...")

        return wrapped

    def is_tarfile(self, file_path):
        try:
            return tarfile.is_tarfile(file_path)
        except Exception as e:
            print(f"Error while checking if file is a tar file: {e}")
            return False


def Schedule(blocknum, blocksize, totalsize):
    global start_time
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: %s" % format_size(speed)
    recv_size = blocknum * blocksize
    f = sys.stdout
    pervent = recv_size / 2650633506
    percent_str = "%.2f%%" % (pervent * 100)
    n = round(pervent * 50)
    s = ("#" * n).ljust(50, "-")
    f.write(percent_str.ljust(8, " ") + "[" + s + "]" + speed_str)
    f.flush()
    time.sleep(0.1)
    f.write("\r")


def format_size(bytes_val):
    try:
        bytes_val = float(bytes_val)
        kb = bytes_val / 1024
    except Exception:
        print("Invalid byte size value")
        return "Error"
    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%.3fG" % (G)
        return "%.3fM" % (M)
    return "%.3fK" % (kb)


def DownloadTif(url, filePath):
    try:
        urllib.request.urlretrieve(url, filePath, reporthook=Schedule)
    except Exception as e:
        print(url + "download failed: " + str(e))


if __name__ == "__main__":

    def parse_args_main():
        parser = argparse.ArgumentParser(description="Downloader for GHCNh data")
        parser.add_argument(
            "--proxy", type=str, default="direct", help="Use 'direct' or 'special' (special needs NWP_HTTP_PROXY)"
        )
        parser.add_argument("--s3", type=str, default=None, help="S3 bucket URL (if needed)")
        parser.add_argument(
            "--worker_num", type=int, default=4, help="Number of threads for parallel downloads"
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["by_station", "by_year"],
            required=True,
            help="Download mode: by_station or by_year",
        )
        parser.add_argument("--start_year", type=int, default=None, help="Start year for 'by_year' mode")
        parser.add_argument("--end_year", type=int, default=None, help="End year for 'by_year' mode")
        return parser.parse_args()

    args = parse_args_main()
    local_root = "./ISD_raw"

    def download_by_station():
        Client = HRES_Client(proxy=args.proxy, s3=args.s3)
        file_path = "./ghcnh-station-list.csv"
        station_name = []
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                station_name.append(row[0])
        print(f"Loaded {len(station_name)} stations from {file_path}.")
        base_url = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-station/"
        threadPool = ThreadPoolExecutor(max_workers=args.worker_num)
        tasks = []
        station_urls = [f"{base_url}GHCNh_{station}_por.psv" for station in station_name]
        station_dir = os.path.join(local_root, "by-station")
        os.makedirs(station_dir, exist_ok=True)
        for station_url in station_urls:
            station_file = station_url.split("/")[-1]
            fp = os.path.join(station_dir, station_file)
            task = threadPool.submit(Client.local_downloader, station_url, fp)
            tasks.append(task)
        for task in as_completed(tasks):
            data = task.result()
            print(f"Task {data} downloaded successfully.")
        threadPool.shutdown(wait=True)
        print("All station files downloaded.")

    def download_by_year():
        Client = HRES_Client(proxy=args.proxy, s3=args.s3)
        file_path = "./ghcnh-station-list.csv"
        station_name = []
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                station_name.append(row[0])
        print(f"Loaded {len(station_name)} stations from {file_path}.")
        base_url = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/"
        threadPool = ThreadPoolExecutor(max_workers=args.worker_num)
        tasks = []
        for year in range(args.start_year, args.end_year + 1):
            print(f"Processing year: {year}")
            year_urls = [f"{base_url}{year}/psv/GHCNh_{station}_{year}.psv" for station in station_name]
            year_dir = os.path.join(f"{local_root}/by-year", str(year))
            os.makedirs(year_dir, exist_ok=True)
            for station_url in year_urls:
                station_file = station_url.split("/")[-1]
                fp = os.path.join(year_dir, station_file)
                task = threadPool.submit(Client.local_downloader, station_url, fp)
                tasks.append(task)
        for task in as_completed(tasks):
            data = task.result()
            print(f"Task {data} downloaded successfully.")
        threadPool.shutdown(wait=True)
        print("All yearly files downloaded.")

    if args.mode == "by_station":
        print("Starting download in by_station mode...")
        download_by_station()
    elif args.mode == "by_year":
        if args.start_year is None or args.end_year is None:
            raise ValueError("For 'by_year' mode, --start_year and --end_year must be specified.")
        print("Starting download in by_year mode...")
        download_by_year()

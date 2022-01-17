from difflib import SequenceMatcher
from itertools import combinations as combs
import os.path
import re
from statistics import mean, pstdev, StatisticsError

import dns.exception
import dns.resolver
import dns.reversename
from geoip2.database import Reader

packagedir = os.path.dirname(__file__)
dbpath = os.path.join(packagedir, '../thirdparty/geoip/GeoLite2-City.mmdb')
city_reader = Reader(dbpath)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def create_resolvers():
    servers = ['1.1.1.1', '8.8.8.8', '208.67.222.222']
    resolvers = []

    for server in servers:
        resolver = dns.resolver.Resolver()
        resolver.nameservers = [server]
        resolvers.append(resolver)

    return resolvers

resolvers = create_resolvers()
cloudflare = dns.resolver.Resolver()
cloudflare.nameservers = ['1.1.1.1']


def dns_features(url):
    domain = re.sub(r'^www\d*', '', url.lower()).lstrip('.')

    ips = set()
    ip_countries = set()
    mx_names = set()
    ns_names = set()
    TTLs = list()

    for resolver in resolvers:
        try:
            a_records = resolver.resolve(domain)
            TTLs.append(a_records.rrset.ttl)
            for record in a_records:
                ips.add(str(record))
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
            dns.exception.Timeout, dns.resolver.NoNameservers, dns.name.LabelTooLong):
            pass
    
        try:
            mx_records = resolver.resolve(domain, 'MX')
            for record in mx_records:
                mx_names.add(str(record))
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
            dns.exception.Timeout, dns.resolver.NoNameservers, dns.name.LabelTooLong):
            pass
    
        try:
            ns_records = resolver.resolve(domain, 'NS')
            for record in ns_records:
                ns_names.add(str(record))
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
            dns.exception.Timeout, dns.resolver.NoNameservers, dns.name.LabelTooLong):
            pass

    # feature 9
    n_ip = len(ips)
    # feature 10
    n_mx = len(mx_names)
    # feature 12
    n_ns = len(ns_names)

    for ip in ips:
        try:
            city_resp = city_reader.city(ip)
            ip_countries.add(city_resp.country.iso_code)
        except:
            pass

    n_countries = len(ip_countries)


    # feature 11
    # use only cloudflare to resolve PTR for faster generation
    ptr_names = set()
    for ip in ips:
        rev_name = dns.reversename.from_address(ip)
        try:
            ptr_records = cloudflare.resolve(rev_name, 'PTR')
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
            dns.exception.Timeout, dns.resolver.NoNameservers):
            continue
        for record in ptr_records:
            ptr_names.add(str(record))
    n_ptr = len(ptr_names)

    # feature 13
    ns_similarity = 1.0
    if len(ns_names) > 2:
        similarities = list()
        all_combs = combs(ns_names, 2)
        for comb in all_combs:
            similarities.append(similarity(*comb))
        ns_similarity = sum(similarities) / len(similarities)
    if n_ip == 0 or len(ns_names) == 0:
        ns_similarity = 0.0

    try:
        mean_TTL = mean(TTLs)
        stdev_TTL = pstdev(TTLs)
    except StatisticsError:
        mean_TTL = 0.0
        stdev_TTL = 0.0

    return (n_ip, n_mx, n_ptr, n_ns, ns_similarity,
            n_countries, mean_TTL, stdev_TTL)


if __name__ == '__main__':
    domain = 'amazon.com'
    print(dns_features(domain))

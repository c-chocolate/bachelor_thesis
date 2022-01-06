from features import Extractor
#domain = 'github.com'
domain = 'www-infosec.ist.osaka-u.ac.jp'
name = True
dns = True
web = True
addweb= True
ext = Extractor(domain, name, dns, web, addweb)

"""
print(ext.get_length())
print(ext.get_n_vowel_chars())
print(ext.get_vowel_ratio())
print(ext.get_n_vowels())
print(ext.get_n_constant_chars())
print(ext.get_n_constants())
print(ext.get_vowel_constant_convs())
print(ext.get_n_nums())
print(ext.get_num_ratio())
print(ext.get_alpha_numer_convs())
print(ext.get_n_other_chars())
print(ext.get_max_consecutive_chars())
print(ext.get_rv())
print(ext.get_entropy())

print(ext.get_n_ip())
print(ext.get_n_mx())
print(ext.get_n_ns())
print(ext.get_n_ptr())
print(ext.get_ns_similarity())
print(ext.get_n_countries())
print(ext.get_mean_TTL())
print(ext.get_stdev_TTL())

print(ext.get_n_labels())
print(ext.get_life_time())
print(ext.get_active_time())
"""

print('CSSのセレクタ数\t\t\t\t', ext.get_n_css_selectors())
print('辞書リストにあるクラス・IDの数\t\t', ext.get_html_id_class_num())
print('辞書リストにあるクラス・IDの割合\t', ext.get_html_id_class_rate())
print('JSのfuntion数\t\t\t\t', ext.get_n_js_function())
print('JSベクトルの平均\t\t\t', ext.get_js_comparison_average())
print('JSベクトルの最大\t\t\t', ext.get_js_comparison_max())
print('JSベクトルの最小\t\t\t', ext.get_js_comparison_min())

#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
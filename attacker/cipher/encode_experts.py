chinese_alphabet = [
    "甲",
    "乙",
    "丙",
    "丁",
    "戊",
    "己",
    "庚",
    "辛",
    "壬",
    "癸",
    "子",
    "丑",
    "寅",
    "卯",
    "辰",
    "巳",
    "午",
    "未",
    "申",
    "酉",
    "戌",
    "亥",
    "天",
    "地",
    "人",
    "黄",
]
english_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                    "u", "v", "w", "x", "y", "z"]


class SelfDefineCipher:
    @staticmethod
    def encode(s):
        s = s.lower()
        ans = ""
        for letter in s:
            try:
                ans += chinese_alphabet[ord(letter.lower()) - 96 - 1]
            except:
                ans += letter
        return ans

    @staticmethod
    def decode(s):
        ans = ""
        for letter in s:
            try:
                position = chinese_alphabet.index(letter)
                ans += english_alphabet[position]
            except:
                ans += letter
        return ans


shift = 3


class CaesarExpert:
    @staticmethod
    def encode(s):
        ans = ""
        for p in s:
            if "a" <= p <= "z":
                ans += chr(ord("a") + (ord(p) - ord("a") + shift) % 26)
            elif "A" <= p <= "Z":
                ans += chr(ord("A") + (ord(p) - ord("A") + shift) % 26)
            else:
                ans += p

        return ans

    @staticmethod
    def decode(s):
        ans = ""
        for p in s:
            if "a" <= p <= "z":
                ans += chr(ord("a") + (ord(p) - ord("a") - shift) % 26)
            elif "A" <= p <= "Z":
                ans += chr(ord("A") + (ord(p) - ord("A") - shift) % 26)
            else:
                ans += p
        return ans


class UnicodeExpert:
    @staticmethod
    def encode(s):
        ans = ""

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("unicode_escape"))
                if len(byte_s) > 8:
                    ans += byte_s[3:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    @staticmethod
    def decode(s):
        ans = bytes(s, encoding="utf8").decode("unicode_escape")
        return ans


class BaseExpert:
    @staticmethod
    def encode(s):
        return s

    @staticmethod
    def decode(s):
        return s


class UTF8Expert:
    @staticmethod
    def encode(s):
        ans = ""

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("utf8"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    @staticmethod
    def decode(s):
        ans = b""
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="utf8")
                s = s[1:]

        ans = ans.decode("utf8")
        return ans


class AsciiExpert:
    @staticmethod
    def encode(s):
        ans = ""

        lines = s.split("\n")
        for line in lines:
            for c in line:
                try:
                    ans += str(ord(c)) + " "
                except:
                    ans += c
            ans += "\n"
        return ans

    @staticmethod
    def decode(s: str):
        ans = ""
        lines = s.split("\n")
        for line in lines:
            cs = line.split()
            for c in cs:
                try:
                    ans += chr(int(c))
                except:
                    ans += c
        return ans


class GBKExpert:
    @staticmethod
    def encode(s):
        ans = ""

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("GBK"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    @staticmethod
    def decode(s):
        ans = b""
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="GBK")
                s = s[1:]

        ans = ans.decode("GBK")
        return ans


class MorseExpert:
    MORSE_CODE_DICT = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----",
        ", ": "--..--",
        ".": ".-.-.-",
        "?": "..--..",
        "/": "-..-.",
        "-": "-....-",
        "(": "-.--.",
        ")": "-.--.-",
    }

    @staticmethod
    def encode(s: str):
        s = s.upper()
        cipher = ""
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                try:
                    if letter != " ":
                        cipher += MorseExpert.MORSE_CODE_DICT[letter] + " "
                    else:
                        cipher += " "
                except:
                    cipher += letter + " "
            cipher += "\n"
        return cipher

    @staticmethod
    def decode(s: str):
        decipher = ""
        citext = ""
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                while True and len(letter):
                    if letter[0] not in ["-", ".", " "]:
                        decipher += letter[0]
                        letter = letter[1:]
                    else:
                        break
                try:
                    if letter != " ":
                        i = 0
                        citext += letter
                    else:
                        i += 1
                        if i == 2:
                            decipher += " "
                        else:
                            decipher += list(MorseExpert.MORSE_CODE_DICT.keys())[

                                list(MorseExpert.MORSE_CODE_DICT.values()).index(citext)]
                            citext = ""
                except:
                    decipher += letter
            decipher += "\n"
        return decipher


class AtbashExpert:
    @staticmethod
    def encode(text: str):
        ans = ""
        N = ord("z") + ord("a")
        for s in text:
            try:
                if s.isalpha():
                    ans += chr(N - ord(s))
                else:
                    ans += s
            except:
                ans += s
        return ans

    @staticmethod
    def decode(text: str):
        ans = ""
        N = ord("z") + ord("a")
        for s in text:
            try:
                if s.isalpha():
                    ans += chr(N - ord(s))
                else:
                    ans += s
            except:
                ans += s
        return ans


ENCODE_EXPERT_DICT = {
    "unchanged": BaseExpert(),
    "baseline": BaseExpert(),
    "caesar": CaesarExpert(),
    "unicode": UnicodeExpert(),
    "morse": MorseExpert(),
    "atbash": AtbashExpert(),
    "utf": UTF8Expert(),
    "ascii": AsciiExpert(),
    "gbk": GBKExpert(),
    "selfdefine": SelfDefineCipher(),
}

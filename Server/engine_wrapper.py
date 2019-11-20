from subprocess import Popen, PIPE


class EngineInstance:

    def __init__(self, path: str, executable) -> None:
        self.engine = Popen(
            f'./{path}/{executable}',
            cwd=f'./{path}',
            universal_newlines=True,
            stdin=PIPE,
            stdout=PIPE
        )
        # await engine ready
        self.send('isready')
        while True:
            text = self.engine.stdout.readline()
            if not text:
                continue
            if 'readyok' in text:
                print('Instance ready!')
                break

    def __del__(self) -> None:
        self.engine.kill()

    def send(self, command: str) -> None:
        self.engine.stdin.write(command + '\n')
        self.engine.stdin.flush()

    def show(self) -> None:
        self.send('d')
        while True:
            text = self.engine.stdout.readline().rstrip()
            print(text)
            # last line is a bunch of numbers
            if '0' in text:
                break

    def setpos(self, posstr: str) -> None:
        self.send(f'position set {posstr}')

    def getbest(self, nodes: int = 1000) -> int:
        self.send(f'getbest n {nodes}')
        ret = self.engine.stdout.readline().strip()
        if ret == 'end of game':
            return -1
        return int(ret.split(' ')[1])

    def geteval(self, nodes: int = 1000) -> str:
        self.send(f'getbest n {nodes}')
        return self.engine.stdout.readline().strip()

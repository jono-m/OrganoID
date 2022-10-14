from multiprocessing import Process, Value, Pipe
from multiprocessing.connection import Connection
import typing
import time


class PipeWriter:
    def __init__(self, connection: Connection):
        self.connection = connection
        self.dataToWrite = ""

    def flush(self):
        self.connection.send(self.dataToWrite)
        self.dataToWrite = ""

    def write(self, data: str):
        self.dataToWrite += data
        return len(data)


class ProcessingWorker:
    def __init__(self):
        self._outputRecv, self._outputSend = Pipe()

        self._outputText = ""

        self._loaded = Value("b", False)

        self._settingsRecvPipe, self._settingsSendPipe = Pipe()
        self._resultsRecvPipe, self._resultsSendPipe = Pipe()

        self._process = Process(target=self.Run,
                                args=(self._loaded, self._settingsRecvPipe, self._resultsSendPipe,
                                      self._outputSend),
                                daemon=True)
        self._process.start()

    def Results(self):
        if self.HasResults():
            return self._resultsRecvPipe.recv()

    def HasResults(self):
        return self._resultsRecvPipe.poll()

    def Process(self, settings: typing.List):
        self._outputText = ""
        self._settingsSendPipe.send(settings)

    @staticmethod
    def Run(loaded, settingsRecvPipe: Connection, resultsSendPipe: Connection,
            outputSendPipe: Connection):
        loaded.value = False
        import sys
        from Core.RunPipeline import RunPipeline
        pipeWriter = PipeWriter(outputSendPipe)
        sys.stdout = pipeWriter
        sys.stderr = pipeWriter

        loaded.value = True
        while True:
            if settingsRecvPipe.poll():
                settings = settingsRecvPipe.recv()
                results = RunPipeline(*settings)
                resultsSendPipe.send(results)
            time.sleep(0.1)

    def ForceStop(self):
        if self._process.is_alive():
            self._process.kill()

    def GetOutputText(self):
        if not self._loaded.value:
            return "Loading OrganoID backend..."

        if self._outputRecv.poll():
            self._outputText += self._outputRecv.recv()

        return self._outputText

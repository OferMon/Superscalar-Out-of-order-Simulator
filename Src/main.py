import re
from copy import deepcopy
import json
import numpy as np
import time
import pandas as pd


class Instruction:

    def __init__(self, memAdd, cmd, oprnds, numOfOps):
        self.memAdd = memAdd
        self.cmd = cmd
        self.oprnds = oprnds
        self.numOfOps = numOfOps

    def __str__(self):
        return str([self.memAdd, self.cmd, self.oprnds, self.numOfOps])


class IDQ:

    def __init__(self, size):
        self.maxSize = size
        self.nextInstIndx = 0
        self.instsQ = []

    def fetch(self, insts):
        # if self.nextInstIndx == len(insts):
        #    return
        self.clearInstsQ()
        lim = min(self.maxSize, len(insts) - self.nextInstIndx)
        for i in range(lim):
            inst = insts[self.nextInstIndx]
            self.instsQ.append(inst)
            if bp.wasBranch:
                bp.history[-1][2] = inst
            if inst.cmd in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'bnez', 'beqz', 'jal', 'jalr', 'j']:
                chckpnt.saveCheck(rob, rat, rs, self.nextInstIndx)
                bp.wasBranch = True
                bp.predict(inst)
            else:
                bp.wasBranch = False
            self.nextInstIndx += 1

    def clearInstsQ(self):
        self.instsQ = []

    def show(self):
        print('IDQ')
        if self.instsQ == []:
            print('IDQ is Empty')
        for i, inst in enumerate(self.instsQ):
            print(f'Instruction {i + 1}- {inst.memAdd}, {inst.cmd}, {inst.oprnds}')
        print('')


class BranchPredictor:

    def __init__(self, direction):
        self.direction = direction
        self.history = []
        self.wasBranch = False

    def predict(self, inst):
        self.history.append([inst, self.direction, None])

    def show(self):
        print('Branch Predictor')
        if self.history == []:
            print('BP is empty')
        else:
            print('Branch Predictor prediction is ', self.direction)
            for i, inst in enumerate(self.history):
                print(f'Row {i + 1}- {inst[0].memAdd}, {inst[0].cmd}, {inst[0].oprnds}')
                if inst[2] is not None:
                    print(f'{inst[2].memAdd}, {inst[2].cmd}, {inst[2].oprnds}')
                else:
                    print('Future instruction is not in the system yet')
        print('')


class RAT:

    def __init__(self, numOfPRF):
        self.FtoAMap = {}
        self.maxSize = numOfPRF
        self.nextPRFAlloc = 0
        self.cMap = {}

    def rename(self, insts):
        copyInsts = deepcopy(insts)
        srcs = []
        for rs in rsArr:
            for inst1 in rs.instsQ:
                if inst1[0].numOfOps == 3 and inst1[0].cmd in ['bltu', 'bgeu']:
                    srcs.append(inst1[0].oprnds)
                elif inst1[0].numOfOps == 3:
                    srcs.append(inst1[0].oprnds[1:])
                elif inst1[0].cmd in ['beq', 'bne', 'blt', 'bge','bltu', 'bgeu', 'bnez', 'beqz']:
                    srcs.append(inst1[0].oprnds)
                else:
                    srcs.append(inst1[0].oprnds[-1])
        for inst in copyInsts:
            if inst.numOfOps < 2:
                continue
            for i, opr in reversed(list(enumerate(inst.oprnds))):
                if opr == 'zero' or 'x' in opr or opr.replace('-', '').isnumeric():
                    continue
                if i != 0 and len(opr) > 2 and hex(int(opr, 16)):
                    continue
                renamedR = self.FtoAMap.get(opr)

                # If I have a map and I'm not the destination
                if i != 0:
                    if renamedR != None:
                        inst.oprnds[i] = renamedR

                # If I'm the destination but rewriting it
                elif inst.cmd not in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'bnez', 'beqz']:
                    if len(self.FtoAMap.keys()) == self.maxSize:
                        print('Not enough PRF allocated, please allocate more!')
                        exit(0)
                    startMap = 'RB' + str(self.nextPRFAlloc)
                    nextMap = 'RB' + str(self.nextPRFAlloc)
                    while True:
                        used = False
                        if bool(self.FtoAMap) == False:
                            break
                        if nextMap not in self.FtoAMap.values():
                            for _, src in enumerate(srcs):
                                if nextMap in src:
                                    used = True
                                    break
                            if used:
                                self.nextPRFAlloc = (self.nextPRFAlloc + 1) % self.maxSize
                                nextMap = 'RB' + str(self.nextPRFAlloc)
                                if nextMap == startMap:
                                    print('Not enough PRF allocated, please allocate more!')
                                    exit(0)
                            else:
                                break
                        else:
                            self.nextPRFAlloc = (self.nextPRFAlloc + 1) % self.maxSize
                            nextMap = 'RB' + str(self.nextPRFAlloc)
                            if nextMap == startMap:
                                print('Not enough PRF allocated, please allocate more!')
                                exit(0)

                    """while nextMap in self.FtoAMap.values():
                        self.nextPRFAlloc = (self.nextPRFAlloc + 1) % self.maxSize
                        nextMap = 'RB' + str(self.nextPRFAlloc)
                        if nextMap == startMap:
                            print('Not enough PRF allocated, please allocate more!')
                            exit(0)"""

                    self.FtoAMap[opr] = nextMap
                    self.cMap[nextMap] = False
                    self.nextPRFAlloc = (self.nextPRFAlloc + 1) % self.maxSize
                    inst.oprnds[i] = self.FtoAMap.get(opr)


                # If I'm the dest but not rewriting it and I have a map
                elif renamedR != None:
                    inst.oprnds[i] = renamedR
        return copyInsts

    def clearR(self, inst):
        oprnds = inst[1].oprnds
        srcs = []
        for rs in rsArr:
            for inst1 in rs.instsQ:
                if inst1[0].numOfOps == 3 and inst1[0].cmd not in ['bltu', 'bgeu']:
                    srcs.append(inst1[0].oprnds[1:])
                elif inst1[0].cmd in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'bnez', 'beqz']:
                    srcs.append(inst1[0].oprnds)
                else:
                    srcs.append(inst1[0].oprnds[-1])
        for opr in oprnds:
            if opr == 'zero' or 'x' in opr or opr.replace('-', '').isnumeric():
                continue
            prf = self.FtoAMap.get(opr)
            if prf == None:
                continue
            clearMap = True
            for src in srcs:
                if prf in src:
                    clearMap = False
                    break
            if inst[1].numOfOps == 3 or inst[1].cmd in ['jal', 'jalr', 'mv']:
                des = inst[1].oprnds[0]
            else:
                return
            if clearMap and prf == inst[-1]:
                self.FtoAMap.pop(opr)

    def show(self):
        print('RAT')
        if self.FtoAMap == {}:
            print('RAT is empty')
        else:
            for i, (k, v) in enumerate(self.FtoAMap.items()):
                print(f'Mapping {i + 1}- {k} : {v}')
        print('')


class ROB:

    def __init__(self, size):
        self.maxSize = size
        self.instsQ = [None] * size
        self.totalRetire = 0
        self.nextPushInstIndx = 0
        self.currRetireInst = 0
        self.countCommit = 0
        self.currSize = 0
        self.instNum = 0
        self.renamedAl = 0

    def commitInst(self, inst, i):
        self.instsQ[i][2] = True
        self.countCommit += 1
        if len(self.instsQ[i]) == 4:
            return self.instsQ[i][3]
        else:
            return None

    def retireInsts(self):
        if self.currSize == 0:
            return
        while True:
            if self.instsQ[self.currRetireInst] is None or self.instsQ[self.currRetireInst][2] == False:
                break
            rat.clearR(self.instsQ[self.currRetireInst])
            self.instsQ[self.currRetireInst] = None
            self.currRetireInst = (self.currRetireInst + 1) % (self.maxSize)
            self.totalRetire += 1
            self.currSize -= 1

    def push(self, insts):
        if self.currSize + len(insts) >= self.maxSize:
            self.retireInsts()
        count = self.maxSize
        for i in self.instsQ:
            if i == None:
                count -= 1
        if count + len(insts) > self.maxSize:
            return False
        for inst in insts:
            # False means the instruction didn't executed yet
            self.instsQ[self.nextPushInstIndx] = [self.nextPushInstIndx, deepcopy(inst), False]
            self.nextPushInstIndx = (self.nextPushInstIndx + 1) % self.maxSize
            self.currSize += 1
            self.instNum += 1
        return True

    def addM(self, insts):
        memAddrss = []
        i = 0
        start = self.renamedAl
        while i != len(insts):
            if self.instsQ[start] is not None:
                memAddrss.append((self.instsQ[start][0], self.instsQ[start][1].memAdd))
            i += 1
            start = (start + 1) % self.maxSize
        self.renamedAl = (self.renamedAl + len(insts)) % self.maxSize
        for i, inst in enumerate(insts):
            for (j, mem) in memAddrss:
                if inst.memAdd == mem:
                    if inst.numOfOps == 3 or inst.cmd in ['jal', 'jalr', 'mv']:
                        des = inst.oprnds[0]
                        self.instsQ[j] += [des]
                        break
                    break

    def show(self):
        print('ROB')
        for i, inst in enumerate(self.instsQ):
            if inst is not None:
                if len(inst) == 4:
                    print(
                        f'Instruction {i + 1}- {inst[1].memAdd}, {inst[1].cmd}, {inst[1].oprnds}, {inst[2]}, {inst[3]}')
                else:
                    print(
                        f'Instruction {i + 1}- {inst[1].memAdd}, {inst[1].cmd}, {inst[1].oprnds}, {inst[2]}')
            else:
                print(f'Empty space')
        print('')


class RS:

    def __init__(self, size):
        self.maxSize = size

        # Each node here- [inst, flag-oprnds ready, flag- executed]
        self.instsQ = []
        self.currSize = 0

    def push(self, insts):
        if self.currSize + len(insts) > self.maxSize:
            return False
        for inst in insts:
            srcs = []
            if inst.numOfOps == 3 and inst.cmd not in ['bltu', 'bgeu']:
                srcs.extend(inst.oprnds[1:])
                numofOp = 2
            elif inst.cmd in ['bltu', 'bgeu']:
                numofOp = 3
                srcs.extend(inst.oprnds)
            elif inst.cmd in ['beq', 'bne', 'blt', 'bge', 'bnez', 'beqz']:
                numofOp = 2
                srcs.extend(inst.oprnds)
            else:
                numofOp = 1
                srcs.append(inst.oprnds[-1])
            instToIns = [inst, [False for i in range(numofOp)], False]
            for i, s in enumerate(srcs):
                reg = rat.cMap.get(s)
                if reg == None or reg == False:
                    continue
                if reg == True:
                    instToIns[1][i] = True
            self.instsQ.append(instToIns)
        self.currSize += len(insts)
        return True

    def retry(self, insts, clock):
        succ = False
        while succ == False:
            clock += 1
            if debugMode == 1:
                print('Clock Cycle number: ', clock - 1)
                idq.show()
                bp.show()
                for indx, rs in enumerate(rsArr):
                    rs.show(indx)
                rob.show()
                rat.show()
                for indx, ex in enumerate(execArr):
                    ex.show(indx)

            for indx, ex in enumerate(execArr):
                ex.freeExUnits(indx)
            for indx, rs in enumerate(rsArr):
                rs.execute(indx)
            global isMoved
            global outSepCount
            if isMoved:
                outSepCount += 1
                isMoved = False
            for indx, rs in enumerate(rsArr):
                succ = rs.push(insts)
                if succ == True:
                    break
        return clock

    def execute(self, indx):
        if len(self.instsQ) == 0 or execArr[indx].freeEx == 0:
            return
        opcodes = [inst[0].cmd for inst in self.instsQ]
        dests = []
        srcs = []
        for inst in self.instsQ:
            if inst[0].numOfOps == 3 and inst[0].cmd not in ['bltu', 'bgeu']:
                dests.append(inst[0].oprnds[0])
                srcs.append(inst[0].oprnds[1:])
            elif inst[0].cmd in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'bnez', 'beqz']:
                srcs.append(inst[0].oprnds)
            elif inst[0].numOfOps == 2:
                dests.append(inst[0].oprnds[0])
                srcs.append(inst[0].oprnds[-1])
            else:
                srcs.append(inst[0].oprnds[-1])
        readys = [inst[1] for inst in self.instsQ]

        for i, src in enumerate(srcs):
            if all(readys[i]) == True or readys[i] == []:
                continue
            allR = [True for _ in range(len(readys[i]))]
            if isinstance(src, list):
                for j, s in enumerate(src):
                    if 'RB' in s and readys[i][j] == False:
                        allR[j] = False
                        break
            else:
                if 'RB' in src and readys[i][0] == False:
                    allR[0] = False
                    break
            if all(allR) == True:
                self.instsQ[i][1] = allR
                continue

        # Check for True-dependence (RaW)
        dep_list = []
        for i, src in enumerate(srcs):
            if isinstance(src, list):
                for dest in dests:
                    if dest in src:
                        dep_list.append(i)
                        break
            else:
                for dest in dests:
                    if dest == src:
                        dep_list.append(i)
                        break

        for j, inst in enumerate(self.instsQ):
            if j in dep_list:
                continue
            if all(inst[1]) == True and inst[2] == False:
                if debugMode == 1:
                    if execArr[indx].freeEx == 0 :
                        if rsType == "Seperate":
                            print('Ex', indx, 'is full!')
                        else:
                            print('Ex are full!')
                        break
                self.instsQ[j][2] = True
                global isMoved
                isMoved = True
                indx2 = self.getIndex(execArr[indx])
                if inst[0].cmd in ['sb', 'sh', 'sw', 'sbu', 'shu', 'lb', 'lh', 'lw', 'lbu', 'lhu']:
                    execArr[indx].inExec.append([j, inst, numOfMemCy - 1, indx2])
                else:
                    execArr[indx].inExec.append([j, inst, 0, indx2])
                execArr[indx].freeEx -= 1

    def getIndex(self, qu):
        for i in range(qu.maxNumOfEx):
            flag = False
            for inst in qu.inExec:
                if i == inst[3]:
                    flag = True
                    break
            if flag == False:
                return i
        return 0

    def updateRS(self, inst):
        if inst[0].numOfOps == 3 or inst[0].cmd in ['jal', 'jalr', 'mv']:
            des = inst[0].oprnds[0]
        else:
            return
        srcs = []
        for inst in self.instsQ:
            if inst[0].numOfOps == 3 and inst[0].cmd not in ['bltu', 'bgeu']:
                srcs.append(inst[0].oprnds[1:])
            elif inst[0].cmd in ['beq', 'bne', 'blt', 'bltu', 'bgeu', 'bge', 'bnez', 'beqz']:
                srcs.append(inst[0].oprnds)
            else:
                srcs.append(inst[0].oprnds[-1])
        for i, src in enumerate(srcs):
            if not isinstance(src, list):
                if des == src:
                    self.instsQ[i][1][src.index(des)] = True
            else:
                for j, s in enumerate(src):
                    if des == s:
                        self.instsQ[i][1][j] = True

    def show(self, indx):
        print('RS', indx)
        if self.currSize == 0:
            print('RS', indx, 'is empty')
        else:
            for i, inst in enumerate(self.instsQ):
                print(f'Instruction {i}- {inst[0].memAdd}, {inst[0].cmd}, {inst[0].oprnds}, {inst[1]}, {inst[2]}')
        print('')


class ExecuteUnits:

    def __init__(self, numOfEx):
        self.maxNumOfEx = numOfEx
        self.freeEx = numOfEx
        self.inExec = []

    def freeExUnits(self, indx):
        if self.freeEx == self.maxNumOfEx:
            return
        memAddrss = []
        i = 0
        start = rob.currRetireInst
        while i != rob.maxSize:
            if rob.instsQ[start] is not None:
                memAddrss.append((rob.instsQ[start][0], rob.instsQ[start][1].memAdd))
            i += 1
            start = (start + 1) % rob.maxSize
        exMemAddrss = [inst[1][0].memAdd for inst in self.inExec]
        rsMemAddrss = [inst[0].memAdd for inst in rsArr[indx].instsQ]
        removeList = []
        for j, inst in enumerate(self.inExec):
            if inst[2] > 0:
                inst[2] -= 1
                continue
            for (i, mem) in memAddrss:
                if inst[1][0].memAdd == mem and rob.instsQ[i][2] == False:
                    reg = rob.commitInst(inst, i)
                    if reg != None:
                        priReg = [k for k, v in rat.FtoAMap.items() if v == reg]
                        if priReg != []:
                            if rob.instsQ[i][1].numOfOps == 3 or rob.instsQ[i][1].cmd in ['jal', 'jalr', 'mv']:
                                des = rob.instsQ[i][1].oprnds[0]
                                if priReg[0] == des:
                                    rat.cMap[reg] = True
                    break
            del rsArr[indx].instsQ[rsMemAddrss.index(inst[1][0].memAdd)]
            del rsMemAddrss[rsMemAddrss.index(inst[1][0].memAdd)]
            rsArr[indx].currSize -= 1
            for rs in rsArr:
                rs.updateRS(inst[1])
            removeList.append(inst[1][0].memAdd)
        for rem in removeList:
            del self.inExec[exMemAddrss.index(rem)]
            del exMemAddrss[exMemAddrss.index(rem)]
            self.freeEx += 1

    def indexes(self, e):
        return e[3]

    def show(self, indx):
        if rsType == 'Seperate':
            print('Execution Unit', indx)
            if self.freeEx == self.maxNumOfEx:
                print('Execution unit', indx, 'is empty')
            else:
                for i, inst in enumerate(self.inExec):
                    print(f'{inst[1][0].memAdd}, {inst[1][0].cmd}, {inst[1][0].oprnds}, {inst[2]}')
        else:
            self.inExec.sort(key=self.indexes)
            execIndx = 0
            for i in range(self.maxNumOfEx):
                if len(self.inExec) > execIndx and self.inExec[execIndx][3] == i:
                    print('Execution Unit', i)

                    print(f'{self.inExec[execIndx][1][0].memAdd}, {self.inExec[execIndx][1][0].cmd},'
                          f' {self.inExec[execIndx][1][0].oprnds}, {self.inExec[execIndx][2]}')
                    execIndx += 1
                else:
                    print('Execution unit', i, 'is empty')
        print('')


class Checkpoint:

    def __init__(self):
        self.rob = None
        self.rat = None
        self.rs = None
        self.nextIDQIndx = None

    def saveCheck(self, rob, rat, rs, nextIDQIndx):
        self.rob = deepcopy(rob)
        self.rat = deepcopy(rat)
        self.rs = deepcopy(rs)
        self.nextIDQIndx = nextIDQIndx

    def loadCheck(self):
        return self.rob, self.rat, self.rs, self.nextIDQIndx


def readTrace(tr):
    instructions = []
    with open(tr + '.trc', 'r') as traceFile:
        lines = traceFile.readlines()
        for line in lines:
            REparts = re.search('0x([0-9a-f]+)\(.*: [0-9a-f]{4,8} ([a-z]+)[ ]+(.*)\n', line)
            if REparts:
                oprs = REparts.group(3)
                REOprd = re.search('([0-9]+)\((.*)\)', oprs)
                if REOprd:
                    oprs = [REparts.group(3).split(',')[0]] + [REOprd.group(1)] + [REOprd.group(2)]
                else:
                    oprs = oprs.split(',')
                inst = Instruction(memAdd=hex(int(REparts.group(1), 16)),
                                   cmd=REparts.group(2),
                                   oprnds=oprs,
                                   numOfOps=len(oprs))
                if inst.cmd in ['sb', 'sh', 'sw', 'sbu', 'shu']:
                    inst.oprnds = inst.oprnds[::-1]
                instructions.append(inst)
    return instructions


def readConfig():
    with open('Configuration.json', 'r') as configFile:
        configuration = json.load(configFile)
    if len(configuration["traces"]) == 0 \
            or configuration["Number Of Pipelines"] not in [1, 2, 3, 4] \
            or configuration["branch Prediction Dir"] not in ["Always Taken", "Always not Taken"] \
            or configuration["Number Of Execution Units"] not in [1, 2, 3, 4] \
            or configuration["reservation Station Type"] not in ["Shared", 'Seperate'] \
            or not isinstance(configuration["reservation Station Size"], int) \
            or configuration["reservation Station Size"] < 1 \
            or not isinstance(configuration["Number Of Memory Cycles"], int) \
            or configuration["Number Of Memory Cycles"] < 1 \
            or not isinstance(configuration["reorder Buffer Size"], int) \
            or configuration["reorder Buffer Size"] < 1 \
            or not isinstance(configuration["Number Of PRF"], int) \
            or configuration["Number Of PRF"] < 1 \
            or configuration["Debug Mode"] not in [0, 1]:
        print("Configuration Error");
        exit(0)
    return configuration


global outSepCount
global isMoved
outSepCount = 0
isMoved = False
if __name__ == "__main__":
    configs = readConfig()
    traces = configs.get('traces')
    for tr in traces:
        outSepCount = 0
        numOfMemCy = configs.get('Number Of Memory Cycles')
        debugMode = configs.get('Debug Mode')
        idq = IDQ(size=configs.get('Number Of Pipelines'))
        bp = BranchPredictor(direction=configs.get('branch Prediction Dir'))
        rob = ROB(size=configs.get('reorder Buffer Size'))
        rat = RAT(numOfPRF=configs.get('Number Of PRF'))
        rs = RS(size=configs.get('reservation Station Size'))
        rsType = configs.get('reservation Station Type')
        numOfEx = configs.get('Number Of Execution Units')
        if rsType == 'Shared':
            rsArr = [rs]
            exec = ExecuteUnits(numOfEx=numOfEx)
            execArr = [exec]
        else:
            exec = ExecuteUnits(numOfEx=1)
            rsArr = [RS(size=configs.get('reservation Station Size')) for _ in range(numOfEx)]
            execArr = [ExecuteUnits(numOfEx=1) for _ in range(numOfEx)]

        chckpnt = Checkpoint()
        instructions = readTrace(tr)

        cyclesTime = []
        currClockCycle = 2
        idq.fetch(insts=instructions)
        while (rob.countCommit < len(instructions)):
            if debugMode == 1:
                print('Clock Cycle number: ', currClockCycle - 1)
                idq.show()
                bp.show()
                for indx, rs in enumerate(rsArr):
                    rs.show(indx)
                rob.show()
                rat.show()
                for indx, ex in enumerate(execArr):
                    ex.show(indx)

            start_time = time.time()
            for indx, ex in enumerate(execArr):
                ex.freeExUnits(indx)
            for indx, rs in enumerate(rsArr):
                rs.execute(indx)

            succ = rob.push(insts=idq.instsQ)
            if succ:
                renamedInsts = rat.rename(insts=idq.instsQ)
                rob.addM(renamedInsts)
                pSucc = False
                for indx, rs in enumerate(rsArr):
                    pSucc = rs.push(insts=renamedInsts)
                    if pSucc == True:
                        break
                if pSucc == False:
                    currClockCycle = rs.retry(insts=renamedInsts, clock=currClockCycle)
                idq.fetch(insts=instructions)

            if isMoved:
                outSepCount += 1
                isMoved = False
            cyclesTime.append(time.time() - start_time)
            currClockCycle += 1
        rob.retireInsts()
        if debugMode == 1:
            print('Clock Cycle number: ', currClockCycle)
            idq.show()
            bp.show()
            for indx, rs in enumerate(rsArr):
                rs.show(indx)
            rob.show()
            rat.show()
            for indx, ex in enumerate(execArr):
                ex.show(indx)

        avgCycleTime = np.average(cyclesTime)
        cpi = currClockCycle / len(instructions)
        ipc = 1 / cpi
        cpuTime = avgCycleTime * cpi * len(instructions)
        ilp = len(instructions) / outSepCount

        totalBranchCmds = len(bp.history)
        truePred = 0
        for b in bp.history:
            if bp.direction == 'Always Taken':
                if b[2] is not None:
                    if hex(int(b[0].oprnds[-1], 16)) == b[2].memAdd:
                        truePred += 1
            else:
                if b[2] is not None:
                    if hex(int(b[0].oprnds[-1], 16)) != b[2].memAdd:
                        truePred += 1

        stats = {'trace': tr,
                 'CpuTime': [round(cpuTime * 10 ** 3)],
                 'IPC': [ipc],
                 'CPI': [cpi],
                 'ILP': [ilp],
                 'Total branch commands': [totalBranchCmds],
                 'True branch predictions (%)': [truePred / totalBranchCmds * 100],
                 }
        df = pd.DataFrame(stats)
        df.to_csv(tr + '.csv')
        """print('cpuTime', round(cpuTime * 10 ** 3), 'msec')
        print('IPC', ipc)
        print('CPI', cpi)
        print('ILP', ilp)
        print('Total branch commands', totalBranchCmds)
        print('True branch predictions percent', truePred / totalBranchCmds * 100, '%')"""

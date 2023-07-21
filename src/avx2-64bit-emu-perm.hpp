#include <cassert>

template <typename vtype>
typename vtype::ymm_t avx2_emu_permute_64bit(typename vtype::ymm_t x, int32_t mask)
{
    switch (mask){
    case 0:
        return vtype::template permutexvar<0>(x);
    case 1:
        return vtype::template permutexvar<1>(x);
    case 2:
        return vtype::template permutexvar<2>(x);
    case 3:
        return vtype::template permutexvar<3>(x);
    case 4:
        return vtype::template permutexvar<4>(x);
    case 5:
        return vtype::template permutexvar<5>(x);
    case 6:
        return vtype::template permutexvar<6>(x);
    case 7:
        return vtype::template permutexvar<7>(x);
    case 8:
        return vtype::template permutexvar<8>(x);
    case 9:
        return vtype::template permutexvar<9>(x);
    case 10:
        return vtype::template permutexvar<10>(x);
    case 11:
        return vtype::template permutexvar<11>(x);
    case 12:
        return vtype::template permutexvar<12>(x);
    case 13:
        return vtype::template permutexvar<13>(x);
    case 14:
        return vtype::template permutexvar<14>(x);
    case 15:
        return vtype::template permutexvar<15>(x);
    case 16:
        return vtype::template permutexvar<16>(x);
    case 17:
        return vtype::template permutexvar<17>(x);
    case 18:
        return vtype::template permutexvar<18>(x);
    case 19:
        return vtype::template permutexvar<19>(x);
    case 20:
        return vtype::template permutexvar<20>(x);
    case 21:
        return vtype::template permutexvar<21>(x);
    case 22:
        return vtype::template permutexvar<22>(x);
    case 23:
        return vtype::template permutexvar<23>(x);
    case 24:
        return vtype::template permutexvar<24>(x);
    case 25:
        return vtype::template permutexvar<25>(x);
    case 26:
        return vtype::template permutexvar<26>(x);
    case 27:
        return vtype::template permutexvar<27>(x);
    case 28:
        return vtype::template permutexvar<28>(x);
    case 29:
        return vtype::template permutexvar<29>(x);
    case 30:
        return vtype::template permutexvar<30>(x);
    case 31:
        return vtype::template permutexvar<31>(x);
    case 32:
        return vtype::template permutexvar<32>(x);
    case 33:
        return vtype::template permutexvar<33>(x);
    case 34:
        return vtype::template permutexvar<34>(x);
    case 35:
        return vtype::template permutexvar<35>(x);
    case 36:
        return vtype::template permutexvar<36>(x);
    case 37:
        return vtype::template permutexvar<37>(x);
    case 38:
        return vtype::template permutexvar<38>(x);
    case 39:
        return vtype::template permutexvar<39>(x);
    case 40:
        return vtype::template permutexvar<40>(x);
    case 41:
        return vtype::template permutexvar<41>(x);
    case 42:
        return vtype::template permutexvar<42>(x);
    case 43:
        return vtype::template permutexvar<43>(x);
    case 44:
        return vtype::template permutexvar<44>(x);
    case 45:
        return vtype::template permutexvar<45>(x);
    case 46:
        return vtype::template permutexvar<46>(x);
    case 47:
        return vtype::template permutexvar<47>(x);
    case 48:
        return vtype::template permutexvar<48>(x);
    case 49:
        return vtype::template permutexvar<49>(x);
    case 50:
        return vtype::template permutexvar<50>(x);
    case 51:
        return vtype::template permutexvar<51>(x);
    case 52:
        return vtype::template permutexvar<52>(x);
    case 53:
        return vtype::template permutexvar<53>(x);
    case 54:
        return vtype::template permutexvar<54>(x);
    case 55:
        return vtype::template permutexvar<55>(x);
    case 56:
        return vtype::template permutexvar<56>(x);
    case 57:
        return vtype::template permutexvar<57>(x);
    case 58:
        return vtype::template permutexvar<58>(x);
    case 59:
        return vtype::template permutexvar<59>(x);
    case 60:
        return vtype::template permutexvar<60>(x);
    case 61:
        return vtype::template permutexvar<61>(x);
    case 62:
        return vtype::template permutexvar<62>(x);
    case 63:
        return vtype::template permutexvar<63>(x);
    case 64:
        return vtype::template permutexvar<64>(x);
    case 65:
        return vtype::template permutexvar<65>(x);
    case 66:
        return vtype::template permutexvar<66>(x);
    case 67:
        return vtype::template permutexvar<67>(x);
    case 68:
        return vtype::template permutexvar<68>(x);
    case 69:
        return vtype::template permutexvar<69>(x);
    case 70:
        return vtype::template permutexvar<70>(x);
    case 71:
        return vtype::template permutexvar<71>(x);
    case 72:
        return vtype::template permutexvar<72>(x);
    case 73:
        return vtype::template permutexvar<73>(x);
    case 74:
        return vtype::template permutexvar<74>(x);
    case 75:
        return vtype::template permutexvar<75>(x);
    case 76:
        return vtype::template permutexvar<76>(x);
    case 77:
        return vtype::template permutexvar<77>(x);
    case 78:
        return vtype::template permutexvar<78>(x);
    case 79:
        return vtype::template permutexvar<79>(x);
    case 80:
        return vtype::template permutexvar<80>(x);
    case 81:
        return vtype::template permutexvar<81>(x);
    case 82:
        return vtype::template permutexvar<82>(x);
    case 83:
        return vtype::template permutexvar<83>(x);
    case 84:
        return vtype::template permutexvar<84>(x);
    case 85:
        return vtype::template permutexvar<85>(x);
    case 86:
        return vtype::template permutexvar<86>(x);
    case 87:
        return vtype::template permutexvar<87>(x);
    case 88:
        return vtype::template permutexvar<88>(x);
    case 89:
        return vtype::template permutexvar<89>(x);
    case 90:
        return vtype::template permutexvar<90>(x);
    case 91:
        return vtype::template permutexvar<91>(x);
    case 92:
        return vtype::template permutexvar<92>(x);
    case 93:
        return vtype::template permutexvar<93>(x);
    case 94:
        return vtype::template permutexvar<94>(x);
    case 95:
        return vtype::template permutexvar<95>(x);
    case 96:
        return vtype::template permutexvar<96>(x);
    case 97:
        return vtype::template permutexvar<97>(x);
    case 98:
        return vtype::template permutexvar<98>(x);
    case 99:
        return vtype::template permutexvar<99>(x);
    case 100:
        return vtype::template permutexvar<100>(x);
    case 101:
        return vtype::template permutexvar<101>(x);
    case 102:
        return vtype::template permutexvar<102>(x);
    case 103:
        return vtype::template permutexvar<103>(x);
    case 104:
        return vtype::template permutexvar<104>(x);
    case 105:
        return vtype::template permutexvar<105>(x);
    case 106:
        return vtype::template permutexvar<106>(x);
    case 107:
        return vtype::template permutexvar<107>(x);
    case 108:
        return vtype::template permutexvar<108>(x);
    case 109:
        return vtype::template permutexvar<109>(x);
    case 110:
        return vtype::template permutexvar<110>(x);
    case 111:
        return vtype::template permutexvar<111>(x);
    case 112:
        return vtype::template permutexvar<112>(x);
    case 113:
        return vtype::template permutexvar<113>(x);
    case 114:
        return vtype::template permutexvar<114>(x);
    case 115:
        return vtype::template permutexvar<115>(x);
    case 116:
        return vtype::template permutexvar<116>(x);
    case 117:
        return vtype::template permutexvar<117>(x);
    case 118:
        return vtype::template permutexvar<118>(x);
    case 119:
        return vtype::template permutexvar<119>(x);
    case 120:
        return vtype::template permutexvar<120>(x);
    case 121:
        return vtype::template permutexvar<121>(x);
    case 122:
        return vtype::template permutexvar<122>(x);
    case 123:
        return vtype::template permutexvar<123>(x);
    case 124:
        return vtype::template permutexvar<124>(x);
    case 125:
        return vtype::template permutexvar<125>(x);
    case 126:
        return vtype::template permutexvar<126>(x);
    case 127:
        return vtype::template permutexvar<127>(x);
    case 128:
        return vtype::template permutexvar<128>(x);
    case 129:
        return vtype::template permutexvar<129>(x);
    case 130:
        return vtype::template permutexvar<130>(x);
    case 131:
        return vtype::template permutexvar<131>(x);
    case 132:
        return vtype::template permutexvar<132>(x);
    case 133:
        return vtype::template permutexvar<133>(x);
    case 134:
        return vtype::template permutexvar<134>(x);
    case 135:
        return vtype::template permutexvar<135>(x);
    case 136:
        return vtype::template permutexvar<136>(x);
    case 137:
        return vtype::template permutexvar<137>(x);
    case 138:
        return vtype::template permutexvar<138>(x);
    case 139:
        return vtype::template permutexvar<139>(x);
    case 140:
        return vtype::template permutexvar<140>(x);
    case 141:
        return vtype::template permutexvar<141>(x);
    case 142:
        return vtype::template permutexvar<142>(x);
    case 143:
        return vtype::template permutexvar<143>(x);
    case 144:
        return vtype::template permutexvar<144>(x);
    case 145:
        return vtype::template permutexvar<145>(x);
    case 146:
        return vtype::template permutexvar<146>(x);
    case 147:
        return vtype::template permutexvar<147>(x);
    case 148:
        return vtype::template permutexvar<148>(x);
    case 149:
        return vtype::template permutexvar<149>(x);
    case 150:
        return vtype::template permutexvar<150>(x);
    case 151:
        return vtype::template permutexvar<151>(x);
    case 152:
        return vtype::template permutexvar<152>(x);
    case 153:
        return vtype::template permutexvar<153>(x);
    case 154:
        return vtype::template permutexvar<154>(x);
    case 155:
        return vtype::template permutexvar<155>(x);
    case 156:
        return vtype::template permutexvar<156>(x);
    case 157:
        return vtype::template permutexvar<157>(x);
    case 158:
        return vtype::template permutexvar<158>(x);
    case 159:
        return vtype::template permutexvar<159>(x);
    case 160:
        return vtype::template permutexvar<160>(x);
    case 161:
        return vtype::template permutexvar<161>(x);
    case 162:
        return vtype::template permutexvar<162>(x);
    case 163:
        return vtype::template permutexvar<163>(x);
    case 164:
        return vtype::template permutexvar<164>(x);
    case 165:
        return vtype::template permutexvar<165>(x);
    case 166:
        return vtype::template permutexvar<166>(x);
    case 167:
        return vtype::template permutexvar<167>(x);
    case 168:
        return vtype::template permutexvar<168>(x);
    case 169:
        return vtype::template permutexvar<169>(x);
    case 170:
        return vtype::template permutexvar<170>(x);
    case 171:
        return vtype::template permutexvar<171>(x);
    case 172:
        return vtype::template permutexvar<172>(x);
    case 173:
        return vtype::template permutexvar<173>(x);
    case 174:
        return vtype::template permutexvar<174>(x);
    case 175:
        return vtype::template permutexvar<175>(x);
    case 176:
        return vtype::template permutexvar<176>(x);
    case 177:
        return vtype::template permutexvar<177>(x);
    case 178:
        return vtype::template permutexvar<178>(x);
    case 179:
        return vtype::template permutexvar<179>(x);
    case 180:
        return vtype::template permutexvar<180>(x);
    case 181:
        return vtype::template permutexvar<181>(x);
    case 182:
        return vtype::template permutexvar<182>(x);
    case 183:
        return vtype::template permutexvar<183>(x);
    case 184:
        return vtype::template permutexvar<184>(x);
    case 185:
        return vtype::template permutexvar<185>(x);
    case 186:
        return vtype::template permutexvar<186>(x);
    case 187:
        return vtype::template permutexvar<187>(x);
    case 188:
        return vtype::template permutexvar<188>(x);
    case 189:
        return vtype::template permutexvar<189>(x);
    case 190:
        return vtype::template permutexvar<190>(x);
    case 191:
        return vtype::template permutexvar<191>(x);
    case 192:
        return vtype::template permutexvar<192>(x);
    case 193:
        return vtype::template permutexvar<193>(x);
    case 194:
        return vtype::template permutexvar<194>(x);
    case 195:
        return vtype::template permutexvar<195>(x);
    case 196:
        return vtype::template permutexvar<196>(x);
    case 197:
        return vtype::template permutexvar<197>(x);
    case 198:
        return vtype::template permutexvar<198>(x);
    case 199:
        return vtype::template permutexvar<199>(x);
    case 200:
        return vtype::template permutexvar<200>(x);
    case 201:
        return vtype::template permutexvar<201>(x);
    case 202:
        return vtype::template permutexvar<202>(x);
    case 203:
        return vtype::template permutexvar<203>(x);
    case 204:
        return vtype::template permutexvar<204>(x);
    case 205:
        return vtype::template permutexvar<205>(x);
    case 206:
        return vtype::template permutexvar<206>(x);
    case 207:
        return vtype::template permutexvar<207>(x);
    case 208:
        return vtype::template permutexvar<208>(x);
    case 209:
        return vtype::template permutexvar<209>(x);
    case 210:
        return vtype::template permutexvar<210>(x);
    case 211:
        return vtype::template permutexvar<211>(x);
    case 212:
        return vtype::template permutexvar<212>(x);
    case 213:
        return vtype::template permutexvar<213>(x);
    case 214:
        return vtype::template permutexvar<214>(x);
    case 215:
        return vtype::template permutexvar<215>(x);
    case 216:
        return vtype::template permutexvar<216>(x);
    case 217:
        return vtype::template permutexvar<217>(x);
    case 218:
        return vtype::template permutexvar<218>(x);
    case 219:
        return vtype::template permutexvar<219>(x);
    case 220:
        return vtype::template permutexvar<220>(x);
    case 221:
        return vtype::template permutexvar<221>(x);
    case 222:
        return vtype::template permutexvar<222>(x);
    case 223:
        return vtype::template permutexvar<223>(x);
    case 224:
        return vtype::template permutexvar<224>(x);
    case 225:
        return vtype::template permutexvar<225>(x);
    case 226:
        return vtype::template permutexvar<226>(x);
    case 227:
        return vtype::template permutexvar<227>(x);
    case 228:
        return vtype::template permutexvar<228>(x);
    case 229:
        return vtype::template permutexvar<229>(x);
    case 230:
        return vtype::template permutexvar<230>(x);
    case 231:
        return vtype::template permutexvar<231>(x);
    case 232:
        return vtype::template permutexvar<232>(x);
    case 233:
        return vtype::template permutexvar<233>(x);
    case 234:
        return vtype::template permutexvar<234>(x);
    case 235:
        return vtype::template permutexvar<235>(x);
    case 236:
        return vtype::template permutexvar<236>(x);
    case 237:
        return vtype::template permutexvar<237>(x);
    case 238:
        return vtype::template permutexvar<238>(x);
    case 239:
        return vtype::template permutexvar<239>(x);
    case 240:
        return vtype::template permutexvar<240>(x);
    case 241:
        return vtype::template permutexvar<241>(x);
    case 242:
        return vtype::template permutexvar<242>(x);
    case 243:
        return vtype::template permutexvar<243>(x);
    case 244:
        return vtype::template permutexvar<244>(x);
    case 245:
        return vtype::template permutexvar<245>(x);
    case 246:
        return vtype::template permutexvar<246>(x);
    case 247:
        return vtype::template permutexvar<247>(x);
    case 248:
        return vtype::template permutexvar<248>(x);
    case 249:
        return vtype::template permutexvar<249>(x);
    case 250:
        return vtype::template permutexvar<250>(x);
    case 251:
        return vtype::template permutexvar<251>(x);
    case 252:
        return vtype::template permutexvar<252>(x);
    case 253:
        return vtype::template permutexvar<253>(x);
    case 254:
        return vtype::template permutexvar<254>(x);
    case 255:
        return vtype::template permutexvar<255>(x);
    default:
        // Should never reach here
        assert(false);
    }
}